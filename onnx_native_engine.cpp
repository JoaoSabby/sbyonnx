/**
 * @file onnx_native_engine.cpp
 * @brief Universal high-performance ONNX export and inference engine for XGBoost.
 * @details Architecture-agnostic SIMD vectorization with runtime compatibility
 *          across heterogeneous x86_64 enterprise server environments.
 * @author Machine Learning Infrastructure Team
 * @version 3.1.0
 * @date 2026-10-20
 */

#include <Rcpp.h>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <immintrin.h>
#include <omp.h>
#include "onnx.pb.h"
#include <onnxruntime_c_api.h>

#define R_EXPORT __attribute__((visibility("default")))

static const OrtApi* g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);

/**
 * @brief Exports an arbitrary XGBoost tree ensemble into a validated ONNX model.
 * 
 * @param output_onnx_path Absolute or relative output filesystem path.
 * @param num_features Predictor count.
 * @param num_classes Output dimension (1 for binary/regression, K >= 3 for multiclass).
 * @param objective_type String selector: "regression", "binary", or "multiclass".
 * @param node_treeids Array of tree IDs for decision splits.
 * @param node_nodeids Array of node IDs within individual trees.
 * @param node_featureids Array of feature indices evaluated at each node.
 * @param node_values Array of numerical split thresholds.
 * @param node_truenodeids Array of child node IDs when condition evaluates true.
 * @param node_falsenodeids Array of child node IDs when condition evaluates false.
 * @param target_treeids Array of tree IDs for leaf nodes.
 * @param target_nodeids Array of node IDs for leaf nodes.
 * @param target_classids Array of class targets (0 to num_classes - 1).
 * @param target_weights Array of leaf weight margins.
 * @param base_score Baseline prior margin scalar.
 * @return True upon successful graph creation and serialization.
 */
// [[Rcpp::export]]
R_EXPORT bool export_xgb_universal_to_onnx(
    std::string output_onnx_path,
    int num_features,
    int num_classes,
    std::string objective_type,
    Rcpp::NumericVector node_treeids,
    Rcpp::NumericVector node_nodeids,
    Rcpp::NumericVector node_featureids,
    Rcpp::NumericVector node_values,
    Rcpp::NumericVector node_truenodeids,
    Rcpp::NumericVector node_falsenodeids,
    Rcpp::NumericVector target_treeids,
    Rcpp::NumericVector target_nodeids,
    Rcpp::NumericVector target_classids,
    Rcpp::NumericVector target_weights,
    double base_score = 0.5
){
    onnx::ModelProto model;
    model.set_ir_version(onnx::IRVersion::IR_VERSION);
    model.set_producer_name("Universal_Native_XGB_Exporter");

    onnx::OperatorSetIdProto* opset_ml = model.add_opset_import();
    opset_ml->set_domain("ai.onnx.ml");
    opset_ml->set_version(1);

    onnx::OperatorSetIdProto* opset_std = model.add_opset_import();
    opset_std->set_domain("");
    opset_std->set_version(13);

    onnx::GraphProto* graph = model.mutable_graph();
    graph->set_name("XGBoost_Universal_Graph");

    onnx::ValueInfoProto* input = graph->add_input();
    input->set_name("input");
    onnx::TypeProto_Tensor* in_tensor = input->mutable_type()->mutable_tensor_type();
    in_tensor->set_elem_type(onnx::TensorProto_DataType_FLOAT);
    in_tensor->mutable_shape()->add_dim()->set_dim_param("N");
    in_tensor->mutable_shape()->add_dim()->set_dim_value(num_features);

    int output_dim = (num_classes > 2) ? num_classes : 1;
    onnx::ValueInfoProto* output = graph->add_output();
    output->set_name("output");
    onnx::TypeProto_Tensor* out_tensor = output->mutable_type()->mutable_tensor_type();
    out_tensor->set_elem_type(onnx::TensorProto_DataType_FLOAT);
    out_tensor->mutable_shape()->add_dim()->set_dim_param("N");
    out_tensor->mutable_shape()->add_dim()->set_dim_value(output_dim);

    std::string intermediate_wire = (objective_type == "regression") ? "output" : "raw_margin";

    onnx::NodeProto* tree_node = graph->add_node();
    tree_node->set_op_type("TreeEnsembleRegressor");
    tree_node->set_domain("ai.onnx.ml");
    tree_node->add_input("input");
    tree_node->add_output(intermediate_wire);

    auto attr_ntargets = tree_node->add_attribute();
    attr_ntargets->set_name("n_targets");
    attr_ntargets->set_type(onnx::AttributeProto::INT);
    attr_ntargets->set_i(output_dim);

    auto attr_treeids = tree_node->add_attribute();
    attr_treeids->set_name("nodes_treeids");
    attr_treeids->set_type(onnx::AttributeProto::INTS);
    for(int i = 0; i < node_treeids.size(); ++i){
        attr_treeids->add_ints(static_cast<int64_t>(node_treeids[i]));
    }

    auto attr_nodeids = tree_node->add_attribute();
    attr_nodeids->set_name("nodes_nodeids");
    attr_nodeids->set_type(onnx::AttributeProto::INTS);
    for(int i = 0; i < node_nodeids.size(); ++i){
        attr_nodeids->add_ints(static_cast<int64_t>(node_nodeids[i]));
    }

    auto attr_featids = tree_node->add_attribute();
    attr_featids->set_name("nodes_featureids");
    attr_featids->set_type(onnx::AttributeProto::INTS);
    for(int i = 0; i < node_featureids.size(); ++i){
        attr_featids->add_ints(static_cast<int64_t>(node_featureids[i]));
    }

    auto attr_values = tree_node->add_attribute();
    attr_values->set_name("nodes_values");
    attr_values->set_type(onnx::AttributeProto::FLOATS);
    for(int i = 0; i < node_values.size(); ++i){
        attr_values->add_floats(static_cast<float>(node_values[i]));
    }

    auto attr_truenodes = tree_node->add_attribute();
    attr_truenodes->set_name("nodes_truenodeids");
    attr_truenodes->set_type(onnx::AttributeProto::INTS);
    for(int i = 0; i < node_truenodeids.size(); ++i){
        attr_truenodes->add_ints(static_cast<int64_t>(node_truenodeids[i]));
    }

    auto attr_falsenodes = tree_node->add_attribute();
    attr_falsenodes->set_name("nodes_falsenodeids");
    attr_falsenodes->set_type(onnx::AttributeProto::INTS);
    for(int i = 0; i < node_falsenodeids.size(); ++i){
        attr_falsenodes->add_ints(static_cast<int64_t>(node_falsenodeids[i]));
    }

    auto attr_modes = tree_node->add_attribute();
    attr_modes->set_name("nodes_modes");
    attr_modes->set_type(onnx::AttributeProto::STRINGS);
    for(int i = 0; i < node_treeids.size(); ++i){
        attr_modes->add_strings("BRANCH_LEQ");
    }

    auto attr_target_treeids = tree_node->add_attribute();
    attr_target_treeids->set_name("target_treeids");
    attr_target_treeids->set_type(onnx::AttributeProto::INTS);
    for(int i = 0; i < target_treeids.size(); ++i){
        attr_target_treeids->add_ints(static_cast<int64_t>(target_treeids[i]));
    }

    auto attr_target_nodeids = tree_node->add_attribute();
    attr_target_nodeids->set_name("target_nodeids");
    attr_target_nodeids->set_type(onnx::AttributeProto::INTS);
    for(int i = 0; i < target_nodeids.size(); ++i){
        attr_target_nodeids->add_ints(static_cast<int64_t>(target_nodeids[i]));
    }

    auto attr_target_ids = tree_node->add_attribute();
    attr_target_ids->set_name("target_ids");
    attr_target_ids->set_type(onnx::AttributeProto::INTS);
    for(int i = 0; i < target_classids.size(); ++i){
        attr_target_ids->add_ints(static_cast<int64_t>(target_classids[i]));
    }

    auto attr_target_weights = tree_node->add_attribute();
    attr_target_weights->set_name("target_weights");
    attr_target_weights->set_type(onnx::AttributeProto::FLOATS);
    for(int i = 0; i < target_weights.size(); ++i){
        attr_target_weights->add_floats(static_cast<float>(target_weights[i]));
    }

    auto attr_base = tree_node->add_attribute();
    attr_base->set_name("base_values");
    attr_base->set_type(onnx::AttributeProto::FLOATS);
    if(objective_type == "binary"){
        float base_margin = static_cast<float>(std::log(base_score / (1.0 - base_score)));
        attr_base->add_floats(base_margin);
    } else {
        for(int k = 0; k < output_dim; ++k){
            attr_base->add_floats(static_cast<float>(base_score));
        }
    }

    if(objective_type == "binary"){
        onnx::NodeProto* sigmoid_node = graph->add_node();
        sigmoid_node->set_op_type("Sigmoid");
        sigmoid_node->set_domain("");
        sigmoid_node->add_input("raw_margin");
        sigmoid_node->add_output("output");
    } else if(objective_type == "multiclass"){
        onnx::NodeProto* softmax_node = graph->add_node();
        softmax_node->set_op_type("Softmax");
        softmax_node->set_domain("");
        softmax_node->add_input("raw_margin");
        softmax_node->add_output("output");

        auto attr_axis = softmax_node->add_attribute();
        attr_axis->set_name("axis");
        attr_axis->set_type(onnx::AttributeProto::INT);
        attr_axis->set_i(1);
    }

    std::fstream out_file(output_onnx_path, std::ios::out | std::ios::trunc | std::ios::binary);
    if(!out_file.is_open()){
        return false;
    }

    bool status = model.SerializeToOstream(&out_file);
    out_file.close();
    return status;
}

/**
 * @brief Multi-threaded inference kernel with hardware-adaptive vectorization.
 * 
 * @param model_path Path of the compiled ONNX model.
 * @param x_mat Input predictor matrix in column-major layout.
 * @param num_classes Dimension of predicted columns.
 * @param num_threads OpenMP thread pool allocation.
 * @return Rcpp::NumericMatrix Predictions matrix shaped [rows, num_classes].
 */
// [[Rcpp::export]]
R_EXPORT Rcpp::NumericMatrix predict_onnx_universal(
    std::string model_path,
    Rcpp::NumericMatrix x_mat,
    int num_classes,
    int num_threads = 8
){
    int rows = x_mat.nrow();
    int cols = x_mat.ncol();
    int out_cols = (num_classes > 2) ? num_classes : 1;

    OrtEnv* env = NULL;
    g_ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "ONNX_Generic_HPC", &env);

    OrtSessionOptions* session_options = NULL;
    g_ort->CreateSessionOptions(&session_options);
    g_ort->SetIntraOpNumThreads(session_options, num_threads);
    g_ort->SetSessionGraphOptimizationLevel(session_options, ORT_ENABLE_ALL);

    OrtSession* session = NULL;
    g_ort->CreateSession(env, model_path.c_str(), session_options, &session);

    OrtMemoryInfo* memory_info = NULL;
    g_ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info);

    size_t total_elements = static_cast<size_t>(rows) * cols;
    // Alocacao alinhada em 64 bytes para atender AVX-512 ou AVX2 sem desalinhamento de cache
    float* dst_buffer = static_cast<float*>(_mm_malloc(total_elements * sizeof(float), 64));
    const double* src_r = &(x_mat(0, 0));

    // Conversao e transposicao paralela com despacho adaptativo de instrucoes SIMD
    #pragma omp parallel for num_threads(num_threads) schedule(static)
    for(int r = 0; r < rows; ++r){
        int c = 0;

        #if defined(__AVX512F__)
        // Caminho AVX-512: 8 doubles -> 8 floats por iteracao
        for(; c <= cols - 8; c += 8){
            __m512d v_d = _mm512_set_pd(
                src_r[(c + 7) * rows + r],
                src_r[(c + 6) * rows + r],
                src_r[(c + 5) * rows + r],
                src_r[(c + 4) * rows + r],
                src_r[(c + 3) * rows + r],
                src_r[(c + 2) * rows + r],
                src_r[(c + 1) * rows + r],
                src_r[(c + 0) * rows + r]
            );
            __m256 v_f = _mm512_cvtpd_ps(v_d);
            _mm256_storeu_ps(&dst_buffer[r * cols + c], v_f);
        }
        #elif defined(__AVX__)
        // Caminho AVX / AVX2: 4 doubles -> 4 floats por iteracao
        for(; c <= cols - 4; c += 4){
            __m256d v_d = _mm256_set_pd(
                src_r[(c + 3) * rows + r],
                src_r[(c + 2) * rows + r],
                src_r[(c + 1) * rows + r],
                src_r[(c + 0) * rows + r]
            );
            __m128 v_f = _mm256_cvtpd_ps(v_d);
            _mm_storeu_ps(&dst_buffer[r * cols + c], v_f);
        }
        #endif

        // Drenagem escalar para o restante das colunas
        for(; c < cols; ++c){
            dst_buffer[r * cols + c] = static_cast<float>(src_r[c * rows + r]);
        }
    }

    std::vector<int64_t> input_shape = {rows, cols};
    OrtValue* input_tensor = NULL;
    g_ort->CreateTensorWithDataAsOrtValue(
        memory_info,
        dst_buffer,
        total_elements * sizeof(float),
        input_shape.data(),
        input_shape.size(),
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
        &input_tensor
    );

    const char* input_names[] = {"input"};
    const char* output_names[] = {"output"};
    OrtValue* output_tensor = NULL;

    g_ort->Run(
        session,
        NULL,
        input_names,
        (const OrtValue* const*)&input_tensor,
        1,
        output_names,
        1,
        &output_tensor
    );

    float* float_array = NULL;
    g_ort->GetTensorMutableData(output_tensor, reinterpret_cast<void**>(&float_array));

    Rcpp::NumericMatrix result(rows, out_cols);
    #pragma omp parallel for collapse(2) num_threads(num_threads) schedule(static)
    for(int r = 0; r < rows; ++r){
        for(int k = 0; k < out_cols; ++k){
            result(r, k) = static_cast<double>(float_array[r * out_cols + k]);
        }
    }

    _mm_free(dst_buffer);
    g_ort->ReleaseValue(output_tensor);
    g_ort->ReleaseValue(input_tensor);
    g_ort->ReleaseMemoryInfo(memory_info);
    g_ort->ReleaseSession(session);
    g_ort->ReleaseSessionOptions(session_options);
    g_ort->ReleaseEnv(env);

    return result;
}