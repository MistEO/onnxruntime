#include "core/optimizer/matmul_bn_fusion.h"
#include "core/graph/graph_utils.h"
#include "core/optimizer/initializer.h"
#include "core/optimizer/utils.h"


namespace onnxruntime
{
    bool MatmulBNFusion::MatchPath(
        const Node& parentNode,
        const gsl::span<std::pair<std::string, std::initializer_list<int>>>& path,
        const Node& childNode) const
    {
        if (path.size() == 0)
        {
            return true;
        }

        if (!graph_utils::IsSupportedOptypeVersionAndDomain(childNode, path[0].first, path[0].second) ||
            childNode.GetExecutionProviderType() != parentNode.GetExecutionProviderType())
        {
            return false;
        }

        return MatchPath(childNode, path.subspan(1), *childNode.OutputNodesBegin());
    }

    /*
    *   Given a MatMul node, it will verify the following pattern.
    *                      MatMul
    *                        |
    *                       / \
    *                      /   \
    *                     /     \
    *               Reshape     Shape
    *                  |          |
    *             Transpose      Cast
    *                  |          |
    *        BatchNormalization  Cast
    *                  |          |
    *              Transpose      |
    *                  |         /
    *                   \       /
    *                    \     /
    *                     \   /
    *                       | 
    *                    Reshape
    * As of writing this fusion, we are being conversative in the pattern because the customer
    * model we are targeting has this exact pattern. Above pattern will evolve in the future 
    * as we tend to add separate fusion to eliminate Transpose around the BatchNormalization, 
    * update the model optimizer script to eliminate adjacent Cast operator, etc.
    * 
    * We have to match the path (MatMul->Shape->Cast->Cast->Reshape) because sub-merging the 
    * BatchNormalization into the MatMul will change MatMul's output and thus we have to make 
    * sure that MatMul's output is not used by any operator to which MatMul's output matters.
    * Other Conditions:
    *   - B tensor of MatMul should be constant.
    *   - scale, B, mean, var tensors of BatchNormalization should be constant.
    *     
    */
    bool MatmulBNFusion::SatisfyCondition(
        const Graph& graph,
        const Node& node,
        const logging::Logger&) const
    {
        if (!graph_utils::IsSupportedOptypeVersionAndDomain(node, "MatMul", { 1, 9, 13 }) ||
            node.GetOutputEdgesCount() != 2)
        {
            return false;
        }

        auto childNodeIterator = node.OutputNodesBegin();
        const Node& firstChildNode = *childNodeIterator;
        ++childNodeIterator;
        const Node& secondChildNode = *childNodeIterator;

        std::vector<std::pair<std::string, std::initializer_list<int>>> firstPath = 
        {{"Reshape", {1, 5}},
         {"Transpose", {1}},
         {"BatchNormalization", {1, 6, 7}},
         {"Transpose", {1}},
         {"Reshape", {1, 5}}};

        std::vector<std::pair<std::string, std::initializer_list<int>>> secondPath =
        {{"Shape", {1}},
         {"Cast", {1, 6}},
         {"Cast", {1, 6}},
         {"Reshape", {1, 5}}};

        if (!(MatchPath(node, firstPath, firstChildNode) ^ MatchPath(node, secondPath, firstChildNode)))
        {
            return false;
        }

        if (!(MatchPath(node, firstPath, secondChildNode) ^ MatchPath(node, secondPath, secondChildNode))) {
            return false;
        }

        
        const auto& batchNormNode = firstChildNode.OpType() == "Reshape" ?
            *firstChildNode.OutputNodesBegin()->OutputNodesBegin() :
            *secondChildNode.OutputNodesBegin()->OutputNodesBegin();
        
        // Check that the appropriate inputs to the Conv and BN nodes are constants.
        if (!graph_utils::NodeArgIsConstant(graph, *node.InputDefs()[1]) ||
            !graph_utils::NodeArgIsConstant(graph, *batchNormNode.InputDefs()[1]) ||
            !graph_utils::NodeArgIsConstant(graph, *batchNormNode.InputDefs()[2]) ||
            !graph_utils::NodeArgIsConstant(graph, *batchNormNode.InputDefs()[3]) ||
            !graph_utils::NodeArgIsConstant(graph, *batchNormNode.InputDefs()[4]))
        {
            return false;
        }

        // First output from BN is required. Others are optional. If any optional outputs exist we can't fuse.
        const auto& output_defs = batchNormNode.OutputDefs();
        if (output_defs.size() > 1) {
            for (size_t i = 1, end = output_defs.size(); i < end; ++i) {
              if (output_defs[i] != nullptr && output_defs[i]->Exists())
                return false;
            }
        }

        if (graph.NodeProducesGraphOutput(node)) {
            return false;
        }

        return true;
    }

    Status MatmulBNFusion::Apply(
        Graph&,
        Node&,
        RewriteRuleEffect&,
        const logging::Logger&) const
    {
        return Status::OK();
    }
}