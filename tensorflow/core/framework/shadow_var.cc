#include "tensorflow/core/framework/shadow_var.h"

namespace tensorflow {
//Some functions for shadows_ (insert shadow, delete shadow, get_shadow)
//When insert, If shadow name is same, the old shadow will be replaced.

void ShadowManager::InsertShadow(int64 global_step, string name, const Tensor& tensor){
  mutex_lock l(mu_);
  // Find shadow according name
  auto iter = shadows_.find(name);
  if (iter == shadows_.end()){
    // Insert a new shadow.
    shadows_.emplace(std::piecewise_construct,
                     std::forward_as_tuple(name),
                     std::forward_as_tuple(global_step, name, tensor));
  } else {
    // Update the existed shadow.
    iter->second.Update(global_step, name, tensor);
  }
}

const ShadowVar ShadowManager::GetShadow(string name){
  mutex_lock l(mu_);
  auto iter = shadows_.find(name);
  if(iter == shadows_.end()){
    // Return an null ShadowVar.
    return ShadowVar(0, "", Tensor());
  }
  return iter->second;
}

// Get all shadows in this worker.
void ShadowManager::GetAllShadowNames(std::vector<string>* all_shadow_names,
                                      std::vector<int64>* all_shadow_steps){
  mutex_lock l(mu_);
  for(auto iter=shadows_.begin(); iter!=shadows_.end(); ++iter){
    all_shadow_names->emplace_back(iter->second.name());
    all_shadow_steps->emplace_back(iter->second.global_step());
  }
}

int ShadowManager::number_shadows(){
  mutex_lock l(mu_);
  return shadows_.size();
}

// The global shadow manager.
ShadowManager g_shadow_manager;

} // end namespace tensorflow