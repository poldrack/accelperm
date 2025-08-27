# FSL Randomise Implementation Pseudocode

This document contains pseudocode extracted from the FSL randomise implementation, providing a detailed guide for implementing GPU-accelerated permutation testing for neuroimaging data.

## Table of Contents

1. [Main Program Flow](#main-program-flow)
2. [Data Initialization](#data-initialization)
3. [Permutation Engine](#permutation-engine)
4. [Statistical Computations](#statistical-computations)
5. [TFCE Implementation](#tfce-implementation)
6. [Multiple Comparison Corrections](#multiple-comparison-corrections)
7. [Output Generation](#output-generation)
8. [Utility Functions](#utility-functions)

---

## Main Program Flow

### Main Function
```pseudocode
FUNCTION main(argc, argv):
    // Parse command line options
    opts = parse_command_line(argc, argv)
    
    // Initialize data structures
    CALL Initialize(opts, mask, data, t_contrasts, design_matrix, f_contrasts, groups, effective_design)
    
    // Check if design needs demeaning
    needs_demean = TRUE
    FOR i = 1 to design_matrix.columns:
        IF abs(design_matrix.column(i).sum()) > 0.0001:
            needs_demean = FALSE
    
    // Handle GLM output if requested
    IF opts.feat_mode OR opts.output_glm:
        CALL output_glm_results(data, design_matrix, t_contrasts)
    
    // Process F-contrasts
    IF f_contrasts is not empty:
        CALL analyze_f_contrast(f_contrasts, t_contrasts, design_matrix, data, mask, groups, opts, effective_design)
    
    // Process T-contrasts
    FOR tstat = 1 to t_contrasts.rows:
        CALL analyze_contrast(t_contrasts.row(tstat), design_matrix, data, mask, groups, tstat, opts, effective_design)
    
    RETURN 0
END FUNCTION
```

---

## Data Initialization

### Initialize Function
```pseudocode
FUNCTION Initialize(opts, mask, data_matrix, t_contrasts, design_matrix, f_contrasts, groups, effective_design):
    // Handle TFCE 2D special case
    IF opts.tfce_2d:
        opts.tfce = TRUE
        opts.tfce_height = 2.0
        opts.tfce_size = 1.0
        opts.tfce_connectivity = 26
    
    // Validate permutation number
    IF opts.n_perm < 0:
        THROW Exception("Randomise requires positive number of permutations")
    
    // Set random seed if provided
    IF opts.random_seed is set:
        seed_random_generator(opts.random_seed)
    
    // Load 4D neuroimaging data
    data_4d = read_volume4d(opts.input_file)
    
    // Handle one-sample t-test case
    IF opts.one_sample:
        design_matrix = ones_matrix(data_4d.time_points, 1)
        t_contrasts = ones_matrix(1, 1)
    ELSE:
        design_matrix = read_vest_file(opts.design_file)
        t_contrasts = read_vest_file(opts.contrast_file)
        IF f_contrasts_file is provided:
            f_contrasts = read_vest_file(opts.f_contrasts_file)
    
    // Set up exchangeability groups
    IF opts.groups_file is provided:
        groups = read_vest_file(opts.groups_file)
    ELSE:
        groups = ones_matrix(design_matrix.rows, 1)
    
    // Validate group assignments
    max_group = groups.maximum()
    FOR i = 1 to max_group:
        IF group i is not assigned to any subject:
            THROW Exception("Block must be assigned to at least one design row")
    
    // Create mask
    IF opts.mask_file is provided:
        mask = read_volume(opts.mask_file)
        VALIDATE mask dimensions match data
    ELSE:
        mask = create_non_constant_mask(data_4d)
    
    // Convert 4D data to matrix format
    IF opts.multivariate_mode:
        data_matrix = reshape_multivariate_data(data_4d)
        mask = create_dummy_mask(data_4d.x_size)
    ELSE:
        data_matrix = data_4d.to_matrix(mask)
        IF opts.demean_data:
            data_matrix = remove_mean(data_matrix)
    
    // Setup voxelwise regressors if specified
    IF opts.voxelwise_ev_numbers AND opts.voxelwise_ev_files:
        setup_voxelwise_design(opts.voxelwise_ev_numbers, opts.voxelwise_ev_files, mask, design_matrix.columns)
    
    RETURN data_matrix, design_matrix, t_contrasts, f_contrasts, mask, groups
END FUNCTION
```

### Non-Constant Mask Creation
```pseudocode
FUNCTION create_non_constant_mask(data_4d, all_ones=FALSE):
    mask = zeros_volume(data_4d.x_size, data_4d.y_size, data_4d.z_size)
    
    IF all_ones:
        mask = ones_volume(data_4d.x_size, data_4d.y_size, data_4d.z_size)
        RETURN mask
    
    FOR z = 0 to data_4d.z_size-1:
        FOR y = 0 to data_4d.y_size-1:
            FOR x = 0 to data_4d.x_size-1:
                FOR t = 1 to data_4d.t_size-1:
                    IF data_4d(x,y,z,t) != data_4d(x,y,z,0):
                        mask(x,y,z) = 1
                        BREAK to next voxel
    
    RETURN mask
END FUNCTION
```

---

## Permutation Engine

### Permuter Class Structure
```pseudocode
CLASS Permuter:
    PROPERTIES:
        is_flipping: BOOLEAN
        is_random: BOOLEAN 
        is_permuting_blocks: BOOLEAN
        n_blocks: INTEGER
        n_subjects: INTEGER
        final_permutation: DOUBLE
        unique_permutations: VECTOR[DOUBLE]
        permuted_labels: VECTOR[COLUMN_VECTOR]
        original_labels: VECTOR[COLUMN_VECTOR]
        original_location: VECTOR[COLUMN_VECTOR]
        previous_permutations: VECTOR[COLUMN_VECTOR]
        true_permutation: COLUMN_VECTOR
        unpermuted_vector: COLUMN_VECTOR
        block_permuter: Permuter*
    
    METHODS:
        create_permutation_scheme()
        next_permutation()
        is_previous_permutation()
        compute_unique_permutations()
        next_shuffle()
        next_flip()
END CLASS
```

### Create Permutation Scheme
```pseudocode
FUNCTION create_permutation_scheme(design, groups, one_non_zero_contrast, required_permutations, detect_null_elements, output_debug, permute_blocks, force_flipping):
    n_blocks = groups.maximum() + 1  // +1 to include "0" block
    n_subjects = design.rows
    
    // Detect null subjects (rows with zero effect)
    IF detect_null_elements:
        FOR row = 1 to n_subjects:
            IF abs(design.row(row).sum()) < 1e-10 AND NOT is_flipping:
                groups(row) = 0  // Mark as null subject
    
    is_permuting_blocks = permute_blocks
    
    // Handle block permutation
    IF is_permuting_blocks:
        block_permuter = new Permuter()
        dummy_blocks = ones_vector(n_blocks - 1)
        effective_block_design = matrix(0, design.columns * design.rows / (n_blocks - 1))
        
        FOR group = 1 to n_blocks-1:
            current_row = empty_row_vector()
            FOR row = 1 to n_subjects:
                IF groups(row) == group:
                    current_row.append(design.row(row))
            effective_block_design.append_row(current_row)
        
        block_permuter.create_permutation_scheme(effective_block_design, dummy_blocks, FALSE, required_permutations, FALSE, output_debug, FALSE, force_flipping)
    
    // Create design labels
    labels = create_design_labels(design.concatenate(groups))
    
    IF force_flipping:
        labels = ones_vector(labels.size)
    
    IF is_permuting_blocks:
        FOR row = 1 to n_subjects:
            labels(row) = row
    
    // Determine if sign-flipping or permutation
    is_flipping = (labels.maximum() == 1 AND one_non_zero_contrast) OR force_flipping
    
    // Initialize permutation blocks
    original_location.resize(n_blocks)
    permuted_labels.resize(n_blocks)
    original_labels.resize(n_blocks)
    
    FOR group = 0 to n_blocks-1:
        member_count = 0
        FOR row = 1 to n_subjects:
            IF groups(row) == group:
                member_count++
        
        original_location[group].resize(member_count)
        permuted_labels[group].resize(member_count)
        
        // Fill in starting locations (backwards)
        FOR row = n_subjects down to 1:
            IF groups(row) == group:
                original_location[group](member_count--) = row
    
    initialize_permutation_blocks(labels, required_permutations)
END FUNCTION
```

### Generate Next Permutation
```pseudocode
FUNCTION next_permutation(permutation_number, print_status, is_storing):
    IF permutation_number != 1 AND print_status:
        PRINT "Starting permutation", permutation_number
    ELSE IF print_status:
        PRINT "Starting permutation", permutation_number, "(Unpermuted data)"
    
    current_labels = permutation_vector()
    new_permutation = empty_vector()
    
    REPEAT:
        IF permutation_number != 1:
            new_permutation = next_permutation(permutation_number)
    WHILE is_random AND is_previous_permutation(new_permutation)
    
    IF is_storing OR is_random:
        previous_permutations.push_back(permutation_vector())
    
    create_true_permutation(permutation_vector(), current_labels, true_permutation)
    RETURN true_permutation
END FUNCTION
```

### Sign Flipping Implementation
```pseudocode
FUNCTION next_flip(flip_vector):
    IF is_random:
        FOR i = 1 to flip_vector.size:
            random_value = random_float(0, 1)
            IF random_value > 0.5:
                flip_vector(i) = 1
            ELSE:
                flip_vector(i) = -1
    ELSE:
        // Systematic enumeration of all 2^n sign patterns
        FOR n = flip_vector.size down to 1:
            IF flip_vector(n) == 1:
                flip_vector(n) = -1
                IF n < flip_vector.size:
                    flip_vector.range(n+1, flip_vector.size) = 1
                RETURN
END FUNCTION
```

### Permutation Shuffle Implementation
```pseudocode
FUNCTION next_shuffle(permutation_vector):
    temp = convert_to_integer_vector(permutation_vector)
    
    IF is_random:
        random_shuffle(temp.begin(), temp.end(), random_generator)
    ELSE:
        next_permutation(temp.begin(), temp.end())
    
    permutation_vector = convert_to_column_vector(temp)
END FUNCTION
```

---

## Statistical Computations

### GLM T-Statistic Calculation
```pseudocode
FUNCTION calculate_t_stat(data, model, t_contrast, estimate, contrast_estimate, residuals, sigma_squared, degrees_of_freedom):
    // Calculate pseudo-inverse of design matrix
    pinv_model = pseudo_inverse(model)
    
    // Parameter estimates (beta coefficients)
    estimate = pinv_model * data
    
    // Calculate residuals
    residuals = data - model * estimate
    
    // Contrast estimates (effect sizes)
    contrast_estimate = t_contrast * estimate
    
    // Variance estimates
    sigma_squared = sum_of_squares_by_rows(residuals) / degrees_of_freedom
    
    // Variance of contrast estimates (varcope)
    varcope = diagonal(t_contrast * pinv_model * pinv_model.transpose() * t_contrast.transpose()) * sigma_squared
    
    // T-statistics
    t_statistics = element_wise_division(contrast_estimate, sqrt(varcope))
    
    RETURN t_statistics
END FUNCTION
```

### GLM F-Statistic Calculation  
```pseudocode
FUNCTION calculate_f_stat(data, model, f_contrast, degrees_of_freedom, rank):
    // model: N_subjects × N_regressors
    // data: N_subjects × N_voxels
    // f_contrast: N_contrasts × N_regressors
    
    pinv_model = pseudo_inverse(model)
    estimate = pinv_model * data
    residuals = data - model * estimate
    
    // Residual sum of squares
    residuals = sum_of_element_wise_products(residuals, residuals) / degrees_of_freedom
    
    // Hypothesis sum of squares
    estimate = pseudo_inverse(f_contrast * pinv_model).transpose() * f_contrast * estimate
    estimate = sum_of_element_wise_products(estimate, estimate) / rank
    
    // F-statistics
    f_statistics = element_wise_division(estimate, residuals)
    
    RETURN f_statistics
END FUNCTION
```

### Multivariate F-Statistic (Pillai's Trace)
```pseudocode
FUNCTION calculate_multivariate_f_stat(model, data, degrees_of_freedom, n_multivariate):
    // model: N_subjects × N_regressors  
    // data: N_subjects × (N_vertices × n_multivariate)
    
    n_vertices = data.columns / n_multivariate
    n_subjects = data.rows
    n_regressors = model.columns
    f_statistics = zeros_matrix(1, n_vertices)
    
    contrast = identity_matrix(n_regressors)  // Test all regressors
    
    FOR vertex = 1 to n_vertices:
        // Extract 3D coordinates for current vertex
        vertex_data = zeros_matrix(n_subjects, 3)
        FOR subject = 1 to n_subjects:
            vertex_data(subject, 1) = data(subject, vertex)
            vertex_data(subject, 2) = data(subject, vertex + n_vertices)
            vertex_data(subject, 3) = data(subject, vertex + 2*n_vertices)
        
        f_statistics(1, vertex) = multivariate_glm_fit(model, vertex_data, contrast, degrees_of_freedom)
    
    RETURN f_statistics
END FUNCTION
```

### Multivariate GLM Fit (Pillai's Trace)
```pseudocode
FUNCTION multivariate_glm_fit(X, Y, contrast, degrees_of_freedom):
    // X: design matrix (N_subjects × N_regressors)
    // Y: data matrix (N_subjects × 3)
    // contrast: contrast matrix (N_contrasts × N_regressors)
    
    // Calculate fitted values
    Y_hat = X * inverse(X.transpose() * X) * X.transpose() * Y
    
    // Calculate residual covariance matrix
    R0 = Y - Y_hat
    R0 = R0.transpose() * R0
    
    // Calculate hypothesis sum of squares matrix
    Y_hat1 = X * contrast.transpose() * inverse(contrast * X.transpose() * X * contrast.transpose()) * contrast * X.transpose() * Y
    R1 = Y - Y_hat1
    R1 = R1.transpose() * R1 - R0
    
    // Calculate Pillai's trace
    g = Y.columns  // number of dependent variables (3 for coordinates)
    p = X.columns  // number of regressors
    N = Y.rows     // sample size
    
    pillai_trace = trace(R1 * inverse(R1 + R0))
    
    // Convert to F-statistic
    s = min(p, g-1)
    t = (abs(p - g - 1) - 1) / 2.0
    u = (N - g - p - 1) / 2.0
    
    F = ((2*u + s + 1) / (2*t + s + 1)) * (pillai_trace / (s - pillai_trace))
    df1 = s * (2*t + s + 1)
    df2 = s * (2*u + s + 1)
    
    degrees_of_freedom[0] = df1
    degrees_of_freedom[1] = df2
    
    RETURN F
END FUNCTION
```

### Variance Smoothing
```pseudocode
FUNCTION smooth_t_stat(input_sigma_squared, mask, smoothed_mask, sigma_mm):
    // Convert variance estimates to volume
    sigma_volume = matrix_to_volume(input_sigma_squared, mask)
    
    // Apply Gaussian smoothing
    sigma_volume = gaussian_smooth(sigma_volume, sigma_mm)
    
    // Normalize by smoothed mask to correct for edge effects
    sigma_volume = element_wise_division(sigma_volume, smoothed_mask)
    
    // Convert back to matrix
    new_sigma_squared = volume_to_matrix(sigma_volume, mask)
    
    // Return smoothing factor
    smoothing_factor = element_wise_division(new_sigma_squared, input_sigma_squared)
    
    RETURN smoothing_factor
END FUNCTION
```

---

## TFCE Implementation

### TFCE Core Algorithm
```pseudocode
FUNCTION tfce(statistical_map, mask, delta, height_power, size_power, connectivity):
    // Convert matrix to volume for processing
    spatial_statistic = matrix_to_volume(statistical_map, mask)
    
    // Apply TFCE transformation
    tfce_enhanced = tfce_volume_processing(spatial_statistic, height_power, size_power, connectivity, 0, delta)
    
    // Convert back to matrix format
    RETURN volume_to_matrix(tfce_enhanced, mask)
END FUNCTION
```

### TFCE Volume Processing (Optimized Version)
```pseudocode
FUNCTION tfce_volume_processing(input_volume, height_power, size_power, connectivity, min_threshold, delta):
    IF delta <= 0:
        delta = input_volume.maximum() / 100.0  // 100 subdivisions of max height
        IF delta <= 0:
            PRINT "Warning: No positive values for TFCE processing"
            RETURN zeros_volume_like(input_volume)
    
    // Initialize output volume
    output_volume = zeros_volume_like(input_volume)
    
    // Get all above-threshold voxel coordinates and values
    above_threshold_voxels = []
    FOR each voxel (x,y,z) in input_volume:
        IF input_volume(x,y,z) > min_threshold:
            above_threshold_voxels.append({x, y, z, input_volume(x,y,z)})
    
    // Sort voxels by intensity value
    sort(above_threshold_voxels by intensity ascending)
    
    // Process thresholds from minimum to maximum
    current_threshold = min_threshold
    processed_voxels = set()
    
    WHILE current_threshold <= input_volume.maximum():
        // Create binary mask for current threshold
        threshold_mask = binarize(input_volume, current_threshold)
        
        // Find connected components
        component_labels, component_sizes = connected_components(threshold_mask, connectivity)
        
        // Enhance each connected component
        FOR each component c with size > 0:
            enhancement_value = power(component_sizes[c], size_power) * power(current_threshold, height_power)
            
            FOR each voxel (x,y,z) where component_labels(x,y,z) == c:
                output_volume(x,y,z) += enhancement_value * delta
        
        current_threshold += delta
    
    RETURN output_volume
END FUNCTION
```

### Connected Components Labeling
```pseudocode
FUNCTION connected_components(binary_volume, connectivity):
    labels_volume = zeros_volume_like(binary_volume)
    component_sizes = []
    current_label = 0
    
    FOR z = 0 to binary_volume.z_size-1:
        FOR y = 0 to binary_volume.y_size-1:
            FOR x = 0 to binary_volume.x_size-1:
                IF binary_volume(x,y,z) > 0 AND labels_volume(x,y,z) == 0:
                    current_label++
                    size = flood_fill_component(binary_volume, labels_volume, x, y, z, current_label, connectivity)
                    component_sizes.append(size)
    
    RETURN labels_volume, component_sizes
END FUNCTION
```

### Flood Fill for Connected Components
```pseudocode
FUNCTION flood_fill_component(binary_volume, labels_volume, start_x, start_y, start_z, label, connectivity):
    queue = [(start_x, start_y, start_z)]
    labels_volume(start_x, start_y, start_z) = label
    component_size = 1
    
    // Define connectivity offsets
    IF connectivity == 6:
        offsets = [(-1,0,0), (1,0,0), (0,-1,0), (0,1,0), (0,0,-1), (0,0,1)]
    ELSE IF connectivity == 18:
        offsets = 6_connectivity + edge_neighbors
    ELSE IF connectivity == 26:
        offsets = all_26_neighbors
    
    WHILE queue is not empty:
        current_x, current_y, current_z = queue.pop_front()
        
        FOR each offset (dx, dy, dz) in offsets:
            neighbor_x = current_x + dx
            neighbor_y = current_y + dy  
            neighbor_z = current_z + dz
            
            IF is_valid_coordinate(neighbor_x, neighbor_y, neighbor_z, binary_volume.dimensions):
                IF binary_volume(neighbor_x, neighbor_y, neighbor_z) > 0 AND labels_volume(neighbor_x, neighbor_y, neighbor_z) == 0:
                    labels_volume(neighbor_x, neighbor_y, neighbor_z) = label
                    queue.push_back((neighbor_x, neighbor_y, neighbor_z))
                    component_size++
    
    RETURN component_size
END FUNCTION
```

### TFCE Statistic Processing (Used in Randomise)
```pseudocode
FUNCTION tfce_statistic(output, input_statistic, mask, tfce_delta, tfce_height, tfce_size, tfce_connectivity, permutation_number, is_f_stat, num_contrasts, degrees_of_freedom, output_permutations, override_delta):
    
    statistical_copy = input_statistic
    
    // Convert F-statistics to Z-statistics if needed
    IF is_f_stat:
        z_statistics = empty_vector()
        dof_vector = fill_vector(input_statistic.size, degrees_of_freedom[0])
        f_to_z_conversion(statistical_copy, num_contrasts, dof_vector, z_statistics)
        statistical_copy = z_statistics.as_row_matrix()
    
    // Auto-calculate delta on first permutation
    IF permutation_number == 1 AND NOT override_delta:
        tfce_delta = statistical_copy.maximum() / 100.0
        IF tfce_delta <= 0:
            PRINT "Warning: No positive values for TFCE processing"
    
    // Apply TFCE if delta is valid
    IF tfce_delta > 0:
        statistical_copy = tfce(statistical_copy, mask, tfce_delta, tfce_height, tfce_size, tfce_connectivity)
    ELSE:
        statistical_copy = zeros_matrix_like(statistical_copy)
    
    // Store results
    output.store(statistical_copy, permutation_number, mask, output_permutations)
    
    RETURN statistical_copy.first_row()
END FUNCTION
```

---

## Multiple Comparison Corrections

### Cluster-Based Thresholding
```pseudocode
FUNCTION cluster_statistic(output, input_statistic, mask, threshold, permutation_number, output_permutations):
    // Convert statistic to spatial volume
    spatial_statistic = matrix_to_volume(input_statistic, mask)
    
    // Binarize at threshold
    binary_volume = binarize(spatial_statistic, threshold)
    
    // Find connected components
    cluster_labels, cluster_sizes = connected_components(binary_volume, CLUSTER_CONNECTIVITY)
    
    // Store cluster size information
    output.store(cluster_labels, cluster_sizes, mask, 1, permutation_number, output_permutations)
END FUNCTION
```

### Cluster-Mass Thresholding
```pseudocode
FUNCTION cluster_mass_statistic(output, input_statistic, mask, threshold, permutation_number, output_permutations):
    // Convert to spatial volumes
    spatial_statistic = matrix_to_volume(input_statistic, mask)
    original_spatial_statistic = spatial_statistic
    
    // Binarize at threshold
    spatial_statistic = binarize(spatial_statistic, threshold)
    
    // Find connected components
    cluster_labels, cluster_sizes = connected_components(spatial_statistic, CLUSTER_CONNECTIVITY)
    
    // Reset cluster sizes and calculate mass instead of extent
    cluster_sizes = zeros_vector(cluster_sizes.size)
    
    FOR z = 0 to mask.z_size-1:
        FOR y = 0 to mask.y_size-1:
            FOR x = 0 to mask.x_size-1:
                IF cluster_labels(x,y,z) > 0:
                    label = cluster_labels(x,y,z)
                    cluster_sizes(label) += original_spatial_statistic(x,y,z)
    
    // Store cluster mass information
    output.store(cluster_labels, cluster_sizes, mask, 1, permutation_number, output_permutations)
END FUNCTION
```

### FDR Correction
```pseudocode
FUNCTION fdr_correction(p_values, q_threshold, conservative_correction=FALSE):
    n_total = p_values.size
    
    // Calculate correction factor
    correction_factor = 1.0
    IF conservative_correction:
        FOR i = 2 to n_total:
            correction_factor += 1.0 / i
    
    // Sort p-values with indices
    sorted_pairs = []
    FOR i = 0 to n_total-1:
        sorted_pairs.append((p_values[i], i))
    
    sort(sorted_pairs by p_value ascending)
    
    // Compute ranks
    ranks = zeros_vector(n_total)
    FOR i = 0 to n_total-1:
        ranks[sorted_pairs[i].index] = i + 1
    
    // Calculate threshold
    q_factor = q_threshold / (correction_factor * n_total)
    p_threshold = 0.0
    
    FOR i = 0 to n_total-1:
        original_index = sorted_pairs[i].index
        IF p_values[original_index] > p_threshold AND p_values[original_index] <= q_factor * ranks[original_index]:
            p_threshold = p_values[original_index]
    
    // Generate adjusted p-values if requested
    adjusted_p_values = zeros_vector(n_total)
    reverse_ranks = zeros_vector(n_total)
    
    FOR i = 0 to n_total-1:
        reverse_ranks[ranks[i] - 1] = i
    
    previous_adjusted = 1.0
    FOR i = n_total-1 down to 0:
        original_index = reverse_ranks[i]
        adjusted_p_values[original_index] = sorted_pairs[i].p_value * correction_factor * n_total / (i + 1)
        adjusted_p_values[original_index] = min(previous_adjusted, adjusted_p_values[original_index])
        previous_adjusted = adjusted_p_values[original_index]
    
    RETURN p_threshold, adjusted_p_values
END FUNCTION
```

---

## Output Generation

### Statistical Output Processing
```pseudocode
FUNCTION output_statistic(input, mask, n_permutations, output_text, output_raw=TRUE, write_critical=TRUE, film_style=FALSE):
    corrected_p_name = film_style ? "p" : "corrp"
    
    // Save raw statistic if requested
    IF output_raw:
        output_volume = matrix_to_volume(input.original_statistic.first_row(), mask)
        save_volume(output_volume, input.output_name.stat_name())
    
    // Get null distribution for first contrast
    distribution = input.maximum_distribution.first_row()
    
    // Save null distribution as text if requested
    IF output_text:
        save_text_file(input.output_name.derived_stat_name(corrected_p_name + ".txt"), distribution.transpose())
    
    // Sort distribution for p-value calculation
    sort(distribution ascending)
    
    // Calculate critical value (95th percentile)
    critical_location = ceiling(0.95 * n_permutations)
    critical_value = distribution[critical_location]
    
    IF write_critical:
        PRINT "Critical Value for:", input.output_name.stat_name(), "is:", critical_value
    
    // Calculate corrected p-values
    n_voxels = input.original_statistic.columns
    corrected_p_values = zeros_matrix(1, n_voxels)
    
    FOR voxel = 1 to n_voxels:
        FOR perm = n_permutations down to 1:
            IF input.original_statistic(1, voxel) > distribution[perm]:
                corrected_p_values(1, voxel) = perm / n_permutations
                BREAK
    
    // Save corrected p-values
    output_volume = matrix_to_volume(corrected_p_values, mask)
    save_volume(output_volume, input.output_name.derived_stat_name(corrected_p_name))
    
    // Save uncorrected p-values if requested
    IF input.output_uncorrected:
        uncorrected_p = input.uncorrected_statistic.first_row() / n_permutations
        output_volume = matrix_to_volume(uncorrected_p, mask)
        save_volume(output_volume, input.output_name.derived_stat_name("p"))
END FUNCTION
```

### Parametric Statistic Storage
```pseudocode
CLASS ParametricStatistic:
    PROPERTIES:
        original_statistic: MATRIX
        uncorrected_statistic: MATRIX  
        maximum_distribution: MATRIX
        sum_stat_matrix: MATRIX
        sum_sample_matrix: MATRIX
        is_averaging: BOOLEAN
        output_uncorrected: BOOLEAN
        storing_uncorrected: BOOLEAN
        output_name: OutputName
    
    FUNCTION setup(n_contrasts, n_permutations, n_voxels, want_average, output_file_name, save_uncorrected=FALSE, store_uncorrected=FALSE):
        output_name = output_file_name
        is_averaging = want_average
        output_uncorrected = save_uncorrected
        storing_uncorrected = save_uncorrected OR store_uncorrected
        
        maximum_distribution.resize(n_contrasts, n_permutations)
        maximum_distribution = 0
        
        IF storing_uncorrected:
            uncorrected_statistic.resize(1, n_voxels)
            uncorrected_statistic = 0
        
        original_statistic.resize(n_contrasts, n_voxels)
        original_statistic = 0
        
        IF is_averaging:
            sum_stat_matrix = original_statistic
            sum_sample_matrix = original_statistic
    
    FUNCTION store(parametric_matrix, permutation_number, mask=NULL, output_raw=FALSE):
        // Store maximum for null distribution
        maximum_distribution.column(permutation_number) = column_wise_maximum(parametric_matrix)
        
        // Store original statistic on first permutation
        IF permutation_number == 1:
            original_statistic = parametric_matrix
        
        // Update uncorrected p-values
        IF storing_uncorrected:
            uncorrected_statistic += greater_than(original_statistic, parametric_matrix)
        
        // Update averaging sums
        IF is_averaging:
            sum_stat_matrix += parametric_matrix
            sum_sample_matrix += element_wise_standard_deviation(parametric_matrix, parametric_matrix)
        
        // Save raw permutation result if requested
        IF output_raw:
            raw_volume = matrix_to_volume(parametric_matrix, mask)
            save_volume(raw_volume, output_name.stat_name("perm" + zero_padded_string(permutation_number, 5)))
END CLASS
```

---

## Utility Functions

### Matrix Operations
```pseudocode
FUNCTION pseudo_inverse(matrix):
    // Compute Moore-Penrose pseudoinverse using SVD
    U, S, V = singular_value_decomposition(matrix)
    
    // Create diagonal matrix with reciprocals of non-zero singular values
    S_inv = zeros_matrix(S.columns, S.rows)
    tolerance = 1e-10 * S.maximum()
    
    FOR i = 1 to S.rows:
        IF S(i,i) > tolerance:
            S_inv(i,i) = 1.0 / S(i,i)
    
    RETURN V * S_inv * U.transpose()
END FUNCTION
```

### Permuted Design Generation
```pseudocode
FUNCTION permuted_design(original_design, permutation_vector, multiply):
    output = original_design
    
    FOR row = 1 to original_design.rows:
        IF multiply:
            // Sign flipping case
            output.row(row) = original_design.row(row) * permutation_vector[row]
        ELSE:
            // Permutation case
            output.row(row) = original_design.row(permutation_vector[row])
    
    RETURN output
END FUNCTION
```

### Contrast Conversion for Confound Handling
```pseudocode
FUNCTION convert_contrast(input_model, input_contrast, input_data, output_model, output_contrast, output_data, mode, debug):
    // mode: 0=Kennedy, 1=Freedman-Lane (default), 2=No confound removal, 3=ter Braak
    
    inverse_contrast = pseudo_inverse(input_contrast)
    W1 = input_model * inverse_contrast  // Interest space
    W2 = input_model - input_model * input_contrast.transpose() * inverse_contrast.transpose()  // Confound space
    
    // Remove null confounds
    original_confounds = W2.columns
    W2_reduced = empty_matrix(W2.rows, 0)
    
    FOR col = 1 to W2.columns:
        temp_column = W2.column(col)
        IF NOT temp_column.is_zero():
            W2_reduced.append_column(temp_column)
    
    W2 = W2_reduced
    confounds_exist = (W2.columns > 0)
    
    IF confounds_exist:
        // SVD to remove numerical rank deficiency
        U, D, V = singular_value_decomposition(W2)
        confound_max = D.maximum()
        
        // Remove very small singular values
        WHILE D[D.columns] < confound_max * 1e-10:
            D = D.sub_matrix(1, D.columns - 1)
        
        W2 = U.columns(1, D.columns)
        
        // Check if confound space is negligible compared to interest space
        U_interest, D_interest, V_interest = singular_value_decomposition(W1)
        interest_max = D_interest.maximum()
        
        IF interest_max > confound_max AND interest_max == (interest_max + confound_max * 10.0):
            confounds_exist = FALSE
            W2 = 0
    
    // Apply appropriate confound handling method
    IF confounds_exist AND (mode == 0 OR mode == 1):
        // Orthogonalize data with respect to confounds
        orthogonalizer = identity_matrix(W2.rows) - W2 * W2.transpose()
        output_data = orthogonalizer * input_data
    ELSE:
        output_data = input_data
    
    // Set up design and contrast
    output_model = W1
    output_contrast = identity_matrix(input_contrast.rows)
    
    IF mode == 0 AND confounds_exist:  // Kennedy method
        output_model = W1 - W2 * W2.transpose() * W1
    
    IF mode == 1 OR mode == 2 OR mode == 3:  // Add confounds to model
        IF confounds_exist:
            nuisance_contrast = zeros_matrix(input_contrast.rows, W2.columns)
            output_contrast.append_columns(nuisance_contrast)
            output_model.append_columns(W2)
        
        IF mode == 3:  // ter Braak: orthogonalize everything
            full_orthogonalizer = identity_matrix(output_model.rows) - output_model * pseudo_inverse(output_model)
            output_data = full_orthogonalizer * input_data
    
    RETURN confounds_exist
END FUNCTION
```

### Design Labels Creation
```pseudocode
FUNCTION create_design_labels(design):
    design_labels = zeros_vector(design.rows)
    known_labels = []
    
    FOR i = 1 to design.rows:
        was_existing_label = FALSE
        
        FOR l = 0 to known_labels.size-1:
            IF design.row(i) == known_labels[l]:
                design_labels[i] = l + 1
                was_existing_label = TRUE
                BREAK
        
        IF NOT was_existing_label:
            known_labels.append(design.row(i))
            design_labels[i] = known_labels.size
    
    RETURN design_labels
END FUNCTION
```

---

## Summary

This pseudocode provides a comprehensive foundation for implementing GPU-accelerated permutation testing. Key implementation priorities for your AccelPerm project should focus on:

1. **Efficient permutation generation** with support for both sign-flipping and full permutation
2. **GPU-optimized GLM computations** for t- and F-statistics
3. **TFCE implementation** with connected components labeling
4. **Memory-efficient storage** of null distributions and results
5. **Exact statistical compatibility** with FSL randomise outputs

The modular design allows for backend-specific optimizations while maintaining the same statistical algorithms and results as the reference FSL implementation.