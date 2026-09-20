# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

if(MSVC)
	add_subdirectory(dll)
else()
	#file(GLOB CPP_SOURCES *.cpp)
	set(CPP_SOURCES ann_exception.cpp aux_utils.cpp index.cpp
        linux_aligned_file_reader.cpp math_utils.cpp memory_mapper.cpp
        partition_and_pq.cpp pq_flash_index.cpp logger.cpp distance.cpp
		 utils.cpp bufann/graph_delta.cpp bufann/index_merger.cpp
        bufann/merge_insert.cpp bufann/inplace_backend.cpp bufann/inplace_graph_ops.cpp
        bufann/bufann_api.cpp bufann/ivf_pq_raw_vector_heap.cpp bufann/ivf_pq_build.cpp
        bufann/ivf_pq_index_file.cpp bufann/ivf_pq_search.cpp bufann/ivf_pq_mutate.cpp
        bufann/ivf_pq_backend.cpp)
	add_library(${PROJECT_NAME} ${CPP_SOURCES})
	add_library(${PROJECT_NAME}_s STATIC ${CPP_SOURCES})
endif()
install()
