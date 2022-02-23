#include "ov_interface.h"
#include <fstream>
#define ORT_API_MANUAL_INIT
#include "core/session/onnxruntime_cxx_api.h"
#include "core/providers/shared_library/provider_api.h"

#if defined (OPENVINO_2022_1)
using Exception = ov::Exception;
#elif defined (OPENVINO_2021_4) 
using Exception = InferenceEngine::Exception;
using WaitMode = InferenceEngine::InferRequest::WaitMode;
#else
using Exception = InferenceEngine::details::InferenceEngineException;
using WaitMode = InferenceEngine::InferRequest::WaitMode;
#endif

namespace onnxruntime {
    namespace openvino_ep {

    const std::string log_tag = "[OpenVINO-EP] ";
    std::shared_ptr<ov_network> ov_core::read_model(const std::string& model) const {
        ov_tensor weights;
        try {
            #if defined (OPENVINO_2022_1)
            return oe.read_model(model, weights);
            #else
            ov_tensor_ptr blob = {nullptr};
            return oe.ReadNetwork(model, blob);
            #endif
            } catch (const Exception& e) {
                ORT_THROW(log_tag + "[OpenVINO-EP] Exception while Reading network: " + std::string(e.what()));
            } catch (...) {
                ORT_THROW(log_tag + "[OpenVINO-EP] Unknown exception while Reading network");
            }
    }
            
    ov_exe_network ov_core::load_network(std::shared_ptr<ov_network>& ie_cnn_network, std::string& hw_target, ov_config config, std::string name) {
        try {
            #if defined (OPENVINO_2022_1)
                auto obj = oe.compile_model(ie_cnn_network, hw_target, config);
                ov_exe_network exe(obj);
                return exe;
            #else 
                auto obj = oe.LoadNetwork(*ie_cnn_network, hw_target, config);
                std::cout << "load network\n";
                ov_exe_network exe(obj);
                return exe;
            #endif     
        } catch (const Exception& e) {
            ORT_THROW(log_tag + " Exception while Loading Network for graph: " + name + e.what());
        } catch (...) {
            ORT_THROW(log_tag + " Exception while Loading Network for graph " + name);
        }    
    }

    ov_exe_network ov_core::import_model(const std::string& compiled_blob, std::string hw_target, std::string name) {
        try {
            #if defined (OPENVINO_2022_1)
            std::ifstream blob_stream_obj(compiled_blob); 
            auto obj = oe.import_model(blob_stream_obj, hw_target, {});
            return ov_exe_network(obj);
            #else
            auto obj = oe.ImportNetwork(compiled_blob, hw_target, {});
            return ov_exe_network(obj);
            #endif
        } catch (Exception &e) {
            ORT_THROW(log_tag + " Exception while Importing Network for graph: " + name + ": " + e.what());
        } catch(...) {
            ORT_THROW(log_tag + " Exception while Importing Network for graph: " + name);
        }
    }

    void ov_core::set_cache(std::string cache_dir_path) {
        #if defined(OPENVINO_2022_1)
        oe.set_property(ov::cache_dir(cache_dir_path));
        #else
        oe.SetConfig({{CONFIG_KEY(CACHE_DIR), cache_dir_path}});
        #endif
    }

    ov_exe_network ov_core::load_network(const std::shared_ptr<const ov_network>& model, const ov_remote_context& context, std::string& name) {
        try {
            #if defined(OPENVINO_2022_1)
            auto obj = oe.compile_model(model, context);
            return ov_exe_network(obj);
            #else
            auto obj = oe.LoadNetwork(*model, context);
            return ov_exe_network(obj);
            #endif
        } catch (const Exception& e) {
            ORT_THROW(log_tag + " Exception while Loading Network for graph: " + name + e.what());
        } catch (...) {
            ORT_THROW(log_tag + " Exception while Loading Network for graph " + name);
        }    
    }

    std::vector<std::string> ov_core::get_available_devices() {
        #if defined (OPENVINO_2022_1)
            auto obj = oe.get_available_devices();
            return obj;
        #else 
            auto obj = oe.GetAvailableDevices();
            return obj;
        #endif
    }
 
    ov_infer_request ov_exe_network::create_infer_request() {
        try {
            #if defined (OPENVINO_2022_1)
                auto infReq = obj.create_infer_request();
                ov_infer_request inf_obj(infReq);
                return inf_obj;
            #else 
                auto infReq = obj.CreateInferRequest(infReq);
                ov_infer_request inf_obj(infReq);
                return inf_obj;
            #endif 
        } catch (const Exception& e) {
            ORT_THROW(log_tag + "Exception while creating InferRequest object: " + e.what());
        } catch (...) {
            ORT_THROW(log_tag + "Exception while creating InferRequest object.");
        }
    }
   
    ov_tensor_ptr ov_infer_request::get_tensor(std::string& input_name) {
        try {
          #if defined (OPENVINO_2022_1)
          auto tobj = ovInfReq.get_tensor(input_name);
          ov_tensor_ptr blob = std::make_shared<ov_tensor>(tobj);
          return blob;
          #else 
          auto blob = infReq.Blob(input_name);
          return blob;
          #endif 
        } catch (const Exception& e) {
          ORT_THROW(log_tag + " Cannot access IE Blob for input: " + input_name + e.what());
        } catch (...) {
          ORT_THROW(log_tag + " Cannot access IE Blob for input: " + input_name);
        }
    }

    void ov_infer_request::set_tensor(ov_tensor& blob, std::string& name) {
        try {
          #if defined(OPENVINO_2022_1)
          ovInfReq.set_tensor(name, blob);
          #else
          infReq.SetBlob(blob, name);
          #endif 
        } catch (const Exception& e) {
          ORT_THROW(log_tag + " Cannot set Remote Blob for output: " + name + e.what());
        } catch (...) {
          ORT_THROW(log_tag + " Cannot set Remote Blob for output: " + name);
        }
    }

    void ov_infer_request::start_async() {
        try {
            #if defined (OPENVINO_2022_1)
            ovInfReq.start_async();
            #else
            infReq.StartAsync();
            #endif 
        } catch (const Exception& e) {
            ORT_THROW(log_tag + " Couldn't start Inference: " + e.what());
        } catch (...) {
            ORT_THROW(log_tag + " Couldn't start Inference");
        }
    }

    void ov_infer_request::wait() {
        try {
            #if defined (OPENVINO_2022_1)
            ovInfReq.wait();
            #else
            infReq.Wait(WaitMode::RESULT_READY); 
            #endif 
        } catch (const Exception& e) {
            ORT_THROW(log_tag + " Exception with completing Inference: " + e.what());
        } catch (...) {
            ORT_THROW(log_tag + " Exception with completing Inference");
        }
    }

    void ov_infer_request::query_status() {
        #if defined (OPENVINO_2022_1)
        std::cout << "ovInfReq.query_state()" << " ";
        #else 
        std::cout << infReq << " "; 
        #endif 
    }

    
    }
}