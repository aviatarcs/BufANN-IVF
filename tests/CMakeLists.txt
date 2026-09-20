# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

option(BUFANN_BUILD_LEGACY_UPDATE_DEMO "Build the standalone BufANN update demo" OFF)

add_executable(build_disk_index build_disk_index.cpp)
if(MSVC)
	target_link_options(build_disk_index PRIVATE /MACHINE:x64 /DEBUG:FULL "/INCLUDE:_tcmalloc")
	target_link_libraries(build_disk_index debug ${CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG}/diskann_dll.lib ${PROJECT_SOURCE_DIR}/dependencies/windows/tcmalloc/libtcmalloc_minimal.lib)
	target_link_libraries(build_disk_index optimized ${CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE}/diskann_dll.lib ${PROJECT_SOURCE_DIR}/dependencies/windows/tcmalloc/libtcmalloc_minimal.lib)
else()
	target_link_libraries(build_disk_index ${PROJECT_NAME} -ltcmalloc aio)
endif()

add_executable(search_disk_index search_disk_index.cpp
    ${PROJECT_SOURCE_DIR}/src/aux_utils.cpp)
if(MSVC)
	target_link_options(search_disk_index PRIVATE /MACHINE:x64 /DEBUG:FULL)
	target_link_libraries(search_disk_index debug ${CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG}/diskann_dll.lib)
	target_link_libraries(search_disk_index optimized ${CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE}/diskann_dll.lib)
else()
	target_link_libraries(search_disk_index ${PROJECT_NAME} aio -ltcmalloc)
endif()

if(BUFANN_BUILD_LEGACY_UPDATE_DEMO)
	add_executable(bufann_update_demo test_update_experiment.cpp)
	if(MSVC)
		target_link_options(bufann_update_demo PRIVATE /MACHINE:x64 /DEBUG:FULL)
		target_link_libraries(bufann_update_demo debug ${CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG}/diskann_dll.lib)
		target_link_libraries(bufann_update_demo optimized ${CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE}/diskann_dll.lib)
	else()
		target_link_libraries(bufann_update_demo ${PROJECT_NAME} aio -ltcmalloc)
	endif()
endif()

# Driver 1: build an index from a data file.
add_executable(bufann_index_builder bufann_index_builder.cpp)
target_link_libraries(bufann_index_builder ${PROJECT_NAME} aio -ltcmalloc)

# Driver 1b: import a canonical DiskANN disk index into inplaceann artifacts.
add_executable(convert_diskann_to_inplaceann convert_diskann_to_inplaceann.cpp)
set_target_properties(convert_diskann_to_inplaceann PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)
target_link_libraries(convert_diskann_to_inplaceann ${PROJECT_NAME} aio -ltcmalloc)

# Driver 1c: fast (file-level streaming) variant of the converter; preferred by
# benchmark/BufANN/common.sh when present.
add_executable(fast_convert_diskann_to_inplaceann fast_convert_diskann_to_inplaceann.cpp)
set_target_properties(fast_convert_diskann_to_inplaceann PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)
target_link_libraries(fast_convert_diskann_to_inplaceann ${PROJECT_NAME} aio -ltcmalloc)

# Driver 2: compute exact ground-truth nearest neighbours (with optional exclusion).
add_executable(bufann_gt_computer bufann_gt_computer.cpp)
target_link_libraries(bufann_gt_computer ${PROJECT_NAME} aio -ltcmalloc)

# Driver 3: use an index — search_only or search_after_update workloads.
add_executable(bufann_driver bufann_driver.cpp)
target_link_libraries(bufann_driver ${PROJECT_NAME} aio -ltcmalloc)

# IVF-PQ raw-vector heap storage primitive test.
add_executable(ivf_pq_raw_vector_heap_test ivf_pq_raw_vector_heap_test.cpp)
target_link_libraries(ivf_pq_raw_vector_heap_test ${PROJECT_NAME} aio -ltcmalloc)

# IVF-PQ build steps (centroid training, cluster assignment) test.
add_executable(ivf_pq_build_test ivf_pq_build_test.cpp)
target_link_libraries(ivf_pq_build_test ${PROJECT_NAME} aio -ltcmalloc)

# Whole-pipeline run at realistic shapes; see the file header for arguments.
add_executable(ivf_pq_scale_test ivf_pq_scale_test.cpp)
target_link_libraries(ivf_pq_scale_test ${PROJECT_NAME} aio -ltcmalloc)

# Recall of ivf_pq_search against ground truth; see the file header.
add_executable(ivf_pq_recall_test ivf_pq_recall_test.cpp)
target_link_libraries(ivf_pq_recall_test ${PROJECT_NAME} aio -ltcmalloc)

# ivf_pq_search against a reference search, brute force, and its guards.
add_executable(ivf_pq_search_test ivf_pq_search_test.cpp)
target_link_libraries(ivf_pq_search_test ${PROJECT_NAME} aio -ltcmalloc)

# ivf_pq_insert against the build's encoders, brute force, and racing searches.
add_executable(ivf_pq_mutate_test ivf_pq_mutate_test.cpp)
target_link_libraries(ivf_pq_mutate_test ${PROJECT_NAME} aio -ltcmalloc)

# IVF-PQ index builder for the benchmark scripts (the driver only loads).
add_executable(ivf_pq_build_index ivf_pq_build_index.cpp)
target_link_libraries(ivf_pq_build_index ${PROJECT_NAME} aio -ltcmalloc)

# IVF-PQ through the public BufANN API.
add_executable(ivf_pq_api_test ivf_pq_api_test.cpp)
target_link_libraries(ivf_pq_api_test ${PROJECT_NAME} aio -ltcmalloc)

# Self-checking tests (non-zero exit on failure); run with `ctest`.
add_test(NAME ivf_pq_raw_vector_heap COMMAND ivf_pq_raw_vector_heap_test)
add_test(NAME ivf_pq_build COMMAND ivf_pq_build_test)
add_test(NAME ivf_pq_scale_smoke COMMAND ivf_pq_scale_test 50000 128 64 32)
add_test(NAME ivf_pq_recall_synthetic COMMAND ivf_pq_recall_test)
add_test(NAME ivf_pq_mutate COMMAND ivf_pq_mutate_test)
add_test(NAME ivf_pq_api COMMAND ivf_pq_api_test)
add_test(NAME ivf_pq_search COMMAND ivf_pq_search_test)
