#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/shadow_var.h"
#include "tensorflow/core/kernels/replication_counter.h"
#include <iostream>

namespace tensorflow{

class GetShadowOp : public OpKernel {
 public:
  explicit GetShadowOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("variable_name", &name_));
  }
  void Compute(OpKernelContext* ctx) override {
    ShadowVar var(g_shadow_manager.GetShadow(name_));
    // std::cout << "var_name : " << name_ << ", g_shadow_manager size : "
    //           << g_shadow_manager.number_shadows() << std::endl;
    if(!var.name().empty()){
      std::cout << "tensor_name: " << var.name()
                << ", NumElements: "<< var.val().NumElements()
                << ", step: " << var.global_step()
                << std::endl;
      
      ctx->set_output(0, var.val());
      return ;
    }
    ctx->SetStatus(errors::NotFound("Can't find the shadow of Variable named : ", name_));
  }

 private:
  string name_; 
};
REGISTER_KERNEL_BUILDER(Name("GetShadow").Device(DEVICE_CPU), GetShadowOp);


class GetAllShadowNamesOp : public OpKernel {
 public:
  explicit GetAllShadowNamesOp(OpKernelConstruction* context) : OpKernel(context) {}
  void Compute(OpKernelContext* ctx) override {
    std::vector<string> all_shadow_names;
    std::vector<int64> all_shadow_steps;
    g_shadow_manager.GetAllShadowNames(&all_shadow_names, &all_shadow_steps);
    // std::cout << "g_shadow_manager size : " << g_shadow_manager.number_shadows()
    //           << ", all_shadow_names.size : " << all_shadow_names.size() << std::endl;
    // for(int i = 0; i < all_shadow_names.size(); ++i){
    //   std::cout << all_shadow_names[i] << ", ";
    // }
    // std::cout << std::endl;

    // for(int i = 0; i < all_shadow_steps.size(); ++i){
    //   std::cout << all_shadow_steps[i] << ", ";
    // }
    // std::cout << std::endl;
    
    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx,
                  ctx->allocate_output(0, TensorShape({}), &output));
    auto num = output->tensor<int32, 0>();
    num() = all_shadow_names.size();

    OP_REQUIRES_OK(ctx,
                ctx->allocate_output(1, TensorShape({all_shadow_names.size()}), &output));
    auto var_names = output->tensor<string, 1>();
    for(int i = 0; i < all_shadow_names.size(); ++i){
      var_names(i) = all_shadow_names[i];
    }

    OP_REQUIRES_OK(ctx,
                ctx->allocate_output(2, TensorShape({all_shadow_steps.size()}), &output));
    auto var_steps = output->tensor<int64, 1>();
    for(int i = 0; i < all_shadow_steps.size(); ++i){
      var_steps(i) = all_shadow_steps[i];
    }
  }
};
REGISTER_KERNEL_BUILDER(Name("GetAllShadowNames").Device(DEVICE_CPU), GetAllShadowNamesOp);



// Get the shadow names and steps in a list.
class GetShadowNamesOp : public OpKernel {
 public:
  explicit GetShadowNamesOp(OpKernelConstruction* context) : OpKernel(context) {}
  void Compute(OpKernelContext* ctx) override {
    
    // Get var list.
    const Tensor& var_list = ctx->input(0);
    const auto& var_list_flat = var_list.flat<string>();
    const int var_list_length = static_cast<int>(var_list.NumElements());

    std::vector<ShadowVar> shadows;
    // Get shadow names and step in var list.
    for (int i = 0; i < var_list_length; ++i){
      string var_name = var_list_flat(i);
      ShadowVar var(g_shadow_manager.GetShadow(var_name));
      // The var name exist in shadow manager.
      if (!var.name().empty()){
        shadows.push_back(std::move(var));
      }
    }

    // for(int i = 0; i < shadows.size(); ++i){
    //   std::cout << "(" << shadows[i]->name() << ", "
    //             << shadows[i]->global_step() << "), ";
    // }
    // std::cout << std::endl;
    
    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx,
                  ctx->allocate_output(0, TensorShape({}), &output));
    auto num = output->tensor<int32, 0>();
    num() = shadows.size();

    Tensor* output_names = nullptr;
    Tensor* output_steps = nullptr;
    OP_REQUIRES_OK(ctx,
                ctx->allocate_output(1, TensorShape({shadows.size()}), &output_names));
    OP_REQUIRES_OK(ctx,
                ctx->allocate_output(2, TensorShape({shadows.size()}), &output_steps));
    auto var_names = output_names->tensor<string, 1>();
    auto var_steps = output_steps->tensor<int64, 1>();
    for(int i = 0; i < shadows.size(); ++i){
      var_names(i) = shadows[i].name();
      var_steps(i) = shadows[i].global_step();
    }

  }
};
REGISTER_KERNEL_BUILDER(Name("GetShadowNames").Device(DEVICE_CPU), GetShadowNamesOp);


int StringToInt(string& str){
  std::stringstream stream(str);
  int temp;
  stream >> temp;
  return temp;
}

string GetDeviceFullName(const string& device_name, bool set_cpu_0 = true){
  // Some default value.
  string job = "ps";
  int replica = 0;
  int task = 0;
  string device_type = "cpu";
  int id = 0;
  for (string spec : str_util::Split(device_name, "/")){
    auto items = str_util::Split(spec, ":");
    if (items.size() == 2 && items[0] == "job")
      job = items[1];
    if (items.size() == 2 && items[0] == "replica")
      replica = StringToInt(items[1]);
    if (items.size() == 2 && items[0] == "task")
      task = StringToInt(items[1]);
    if (items.size() == 3 && items[0] == "device" && !set_cpu_0){
      device_type = items[1];
      id = StringToInt(items[2]);
    }
  }
  return DeviceNameUtils::FullName(job, replica, task, device_type, id);
}


string GetRendezvousKeyPrefix(const string& send_device,
                                    const string& recv_device,
                                    const uint64 send_device_incarnation,
                                    const string& tensor_name) {
  return strings::StrCat(send_device, ";",
                        strings::FpToString(send_device_incarnation), ";",
                        recv_device, ";", tensor_name);
}

void GetRendezvousKey(const string& key_prefix,
                            const FrameAndIter& frame_iter, string* key) {
  key->clear();
  strings::StrAppend(key, key_prefix, ";", frame_iter.frame_id, ":",
                    frame_iter.iter_id);
}

class SendReplicationOp : public AsyncOpKernel {
 public:
  explicit SendReplicationOp(OpKernelConstruction* context) : AsyncOpKernel(context) {
    // OP_REQUIRES_OK(context, context->GetAttr("send_device", &send_device_));
    OP_REQUIRES_OK(context, context->GetAttr("recv_device", &recv_device_));
    OP_REQUIRES_OK(context, context->GetAttr("variable_name", &variable_name_));

    send_device_ = context->device()->name();
    recv_device_ = GetDeviceFullName(recv_device_);
    // std::cout << "send_device is : " << send_device_
    //           << "recv_device is : " << recv_device_ << std::endl;
    string key_prefix = GetRendezvousKeyPrefix(send_device_, recv_device_,
                                        11111, variable_name_);
    GetRendezvousKey(key_prefix, {0, 0}, &parsed_key_.buf_);
    OP_REQUIRES_OK(context, Rendezvous::ParseKey(parsed_key_.buf_, &parsed_key_));
  }

  void ComputeAsync(OpKernelContext* ctx, DoneCallback done) override {
    Rendezvous::Args args;
    args.device_context = ctx->op_device_context();
    args.alloc_attrs = ctx->input_alloc_attr(0);
    ctx->rendezvous()->SendReplicationAsync(parsed_key_,
                                            args,
                                            ctx->input(1).scalar<int64>()(),
                                            variable_name_,
                                            ctx->input(0),
                                            [ctx, done](Status s){
                                              ctx->SetStatus(s);
                                              done();
                                            });
  }
 private:
  string variable_name_;
  string send_device_;
  string recv_device_;
  Rendezvous::ParsedKey parsed_key_;


};
REGISTER_KERNEL_BUILDER(Name("SendReplication").Device(DEVICE_CPU), SendReplicationOp);


class SendReplicationV2Op : public AsyncOpKernel {
 public:
  explicit SendReplicationV2Op(OpKernelConstruction* context) : AsyncOpKernel(context) {

    OP_REQUIRES_OK(context, context->GetAttr("tensor_name", &tensor_name_));
    OP_REQUIRES_OK(context, context->GetAttr("worker_num", &worker_num_));
    OP_REQUIRES_OK(context, context->GetAttr("ps_num", &ps_num_));
    OP_REQUIRES_OK(context, context->GetAttr("k", &k_));
    response_num_ = 0;
  }
  
  void ComputeAsync(OpKernelContext* ctx, DoneCallback done) override {
    const Tensor& tensor = ctx->input(0);
    const Tensor& global_step = ctx->input(1);
    // We increment global_step here.
    const int64 global_step_scalar = global_step.scalar<int64>()();

    bool send_flag  = false;
    int worker_repl_num = 0;
    ReplicationCounter* counter = g_replication_counter_manager.GetOrCreateCounter(tensor_name_);
    {
      mutex_lock l(counter->mu);
      // Increment the send counter.
      ++(counter->send_counter);
      if(counter->send_counter >= worker_num_){
        send_flag = true;
        worker_repl_num = counter->pull_worker_set.size();
        // std::cout << "tensorflow::SET  ";
        // for (string worker_str : counter->pull_worker_set){
        //   std::cout << worker_str << ",";
        // }
        // std::cout << std::endl;
        counter->ResetUnlock();
      }
    }

    if (send_flag){  
      // Parse send device
      string send_device = ctx->device()->name();
      auto items = str_util::Split(send_device, "/");
      string job = str_util::Split(items[1], ":")[1];
      int task = StringToInt(str_util::Split(items[3], ":")[1]);

      // Check the send device is "ps" or not
      if (job != "ps"){
        ctx->SetStatus(
          errors::Internal("The SendReplication Op should on PS, but current worker is : ", job));
        done();
        return ;
      }
      // Compute the number of recv devices
      num_recv_devices_ = k_ -1 - worker_repl_num;

      // std::cout << "tensorflow::INFO  "
      //           << "tensor_name:" << tensor_name_
      //           << ", worker_num:" << worker_num_
      //           << ", ps_num:" << ps_num_
      //           << ", k:" << k_
      //           << ", num_recv_devices_:" << num_recv_devices_
      //           << ", worker_repl_num:" << worker_repl_num
      //           << std::endl;

      // num_recv_devices should < ps_num_
      // Because the original PS is not included
      if (num_recv_devices_ >= ps_num_){
        ctx->SetStatus(
          errors::Internal("The k value is so large that ps number is not enough"));
        done();
        return ;
      }

      // TODO: need to change
      if (num_recv_devices_ <= 0){
        ctx->SetStatus(status_);
        done();
        // std::cout << "!!!!!!!!!!!!!!!!!!!!!!num recv devices is :" << num_recv_devices_ << std::endl;
        return ;
      }
      

      // Send replications
      response_num_ = 0;
      for (int i = 1; i <= num_recv_devices_; ++i){
        string recv_device = DeviceNameUtils::FullName(job, 0, (task+i)%ps_num_, "cpu", 0);
        
        // std::cout << "tensorflow::SEND  "
        //           << "tensor_name: " << tensor_name_
        //           << ", send_device: " << send_device
        //           << ", recv_device: " << recv_device << std::endl;

        string key_prefix = GetRendezvousKeyPrefix(send_device, recv_device,
                                      11111, tensor_name_);
        Rendezvous::ParsedKey parsed_key;
        GetRendezvousKey(key_prefix, {0, 0}, &parsed_key.buf_);
        OP_REQUIRES_OK(ctx, Rendezvous::ParseKey(parsed_key.buf_, &parsed_key));

        Rendezvous::Args args;
        args.device_context = ctx->op_device_context();
        args.alloc_attrs = ctx->input_alloc_attr(0);

        ctx->rendezvous()->SendReplicationAsync(
            parsed_key, args, global_step_scalar, tensor_name_, tensor,
            [ctx, done, this](Status s){
              if (!s.ok()){
                status_.Update(s);
                ctx->SetStatus(status_);
                done();
                return ;
              }
              {
                mutex_lock l(mu_);
                ++response_num_;
                if (response_num_ >= num_recv_devices_){
                  ctx->SetStatus(status_);
                  done();
                  // std::cout << "done!!!!!" << std::endl;
                }
              }
            });
      }
      
    } else {
      // do nothing, return.
      // std::cout << "tensorflow::INCREMENT  "
      //           << "tensor_name : " << tensor_name_ 
      //           << ", just increment send counter: " << counter->send_counter
      //           << std::endl;
      done();
    }

  }
 private:
  mutex mu_;
  string tensor_name_;
  Status status_;
  int response_num_;
  int num_recv_devices_;
  int worker_num_;
  int ps_num_;
  int k_;
};
REGISTER_KERNEL_BUILDER(Name("SendReplicationV2").Device(DEVICE_CPU), SendReplicationV2Op);


class SendReplicationV3Op : public AsyncOpKernel {
 public:
  explicit SendReplicationV3Op(OpKernelConstruction* context) : AsyncOpKernel(context) {

    OP_REQUIRES_OK(context, context->GetAttr("tensor_name", &tensor_name_));
    OP_REQUIRES_OK(context, context->GetAttr("worker_num", &worker_num_));
    OP_REQUIRES_OK(context, context->GetAttr("ps_num", &ps_num_));
    OP_REQUIRES_OK(context, context->GetAttr("k", &k_));
    response_num_ = 0;
  }

  void ComputeAsync(OpKernelContext* ctx, DoneCallback done) override {
    const Tensor& tensor = ctx->input(0);
    const Tensor& global_step = ctx->input(1);
    // We increment global_step here.
    const int64 global_step_scalar = global_step.scalar<int64>()();

    bool send_flag  = false;
    ReplicationCounter* counter = g_replication_counter_manager.GetOrCreateCounter(tensor_name_);
    {
      mutex_lock l(counter->mu);
      // Increment the send counter.
      ++(counter->send_counter);
      if(counter->send_counter >= worker_num_){
        send_flag = true;
        counter->ResetUnlock();
      }
    }

    if (send_flag) {
      // Parse send device
      string send_device = ctx->device()->name();
      auto items = str_util::Split(send_device, "/");
      string job = str_util::Split(items[1], ":")[1];
      int task = StringToInt(str_util::Split(items[3], ":")[1]);

      // Check the send device is "ps" or not
      if (job != "ps"){
        ctx->SetStatus(
          errors::Internal("The SendReplicationV3 Op should on PS, but current worker is : ", job));
        done();
        return ;
      }
      // Compute the number of recv devices
      num_recv_devices_ = k_ - 1;

      // std::cout << "tensorflow::INFO  "
      //           << "tensor_name:" << tensor_name_
      //           << ", worker_num:" << worker_num_
      //           << ", ps_num:" << ps_num_
      //           << ", k:" << k_
      //           << ", num_recv_devices_:" << num_recv_devices_
      //           << std::endl;

      // num_recv_devices should < ps_num_
      // Because the original PS is not included
      if (num_recv_devices_ >= ps_num_){
        ctx->SetStatus(
          errors::Internal("The k value is so large that ps number is not enough"));
        done();
        return ;
      }

      // TODO: need to change
      if (num_recv_devices_ <= 0){
        ctx->SetStatus(status_);
        done();
        // std::cout << "!!!!!!!!!!!!!!!!!!!!!!num recv devices is :" << num_recv_devices_ << std::endl;
        return ;
      }
      

      // Send replications
      response_num_ = 0;
      for (int i = 1; i <= num_recv_devices_; ++i){
        string recv_device = DeviceNameUtils::FullName(job, 0, (task+i)%ps_num_, "cpu", 0);
        
        // std::cout << "tensorflow::SEND  "
        //           << "tensor_name: " << tensor_name_
        //           << ", send_device: " << send_device
        //           << ", recv_device: " << recv_device << std::endl;

        string key_prefix = GetRendezvousKeyPrefix(send_device, recv_device,
                                      11111, tensor_name_);
        Rendezvous::ParsedKey parsed_key;
        GetRendezvousKey(key_prefix, {0, 0}, &parsed_key.buf_);
        OP_REQUIRES_OK(ctx, Rendezvous::ParseKey(parsed_key.buf_, &parsed_key));

        Rendezvous::Args args;
        args.device_context = ctx->op_device_context();
        args.alloc_attrs = ctx->input_alloc_attr(0);

        ctx->rendezvous()->SendReplicationAsync(
            parsed_key, args, global_step_scalar, tensor_name_, tensor,
            [ctx, done, this](Status s){
              if (!s.ok()){
                status_.Update(s);
                ctx->SetStatus(status_);
                done();
                return ;
              }
              {
                mutex_lock l(mu_);
                ++response_num_;
                if (response_num_ >= num_recv_devices_){
                  ctx->SetStatus(status_);
                  done();
                  // std::cout << "done!!!!!" << std::endl;
                }
              }
            });
      }
    } else {
      // do nothing, return.
      // std::cout << "tensorflow::INCREMENT  "
      //           << "tensor_name : " << tensor_name_ 
      //           << ", just increment send counter: " << counter->send_counter
      //           << std::endl;
      done();
    }
  }
 private:
  mutex mu_;
  string tensor_name_;
  Status status_;
  int response_num_;
  int num_recv_devices_;
  int worker_num_;
  int ps_num_;
  int k_;


};
REGISTER_KERNEL_BUILDER(Name("SendReplicationV3").Device(DEVICE_CPU), SendReplicationV3Op);

}
