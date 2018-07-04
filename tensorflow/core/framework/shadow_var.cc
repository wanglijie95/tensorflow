#include "tensorflow/core/framework/shadow_var.h"

namespace tensorflow {
//Some functions for shadows_ (insert shadow, delete shadow, get_shadow)
//When insert, If shadow name is same, the old shadow will be replaced.
void ShadowManager::InsertShadow(ShadowVar* shadow){
  mutex_lock l(mu_);
  if (GetShadowUnlock(shadow->name()) == nullptr){
    shadows_[shadow->name()] = shadow;
  } else {
    ShadowVar* old_var = shadows_[shadow->name()];
    shadows_[shadow->name()] = shadow;
    delete old_var;
  }
}

void ShadowManager::InsertShadow(int64 global_step, string name, const Tensor& tensor){
  mutex_lock l(mu_);
  ShadowVar* shadow = new ShadowVar(global_step, name, tensor);
  if (GetShadowUnlock(shadow->name()) == nullptr){
    shadows_[shadow->name()] = shadow;
  } else {
    ShadowVar* old_var = shadows_[shadow->name()];
    shadows_[shadow->name()] = shadow;
    delete old_var;
  }
}

void ShadowManager::DeleteShadow(string name){
  mutex_lock l(mu_);
  ShadowVar* old_var = GetShadowUnlock(name);
  if (old_var != nullptr){
    shadows_.erase(name);
    delete old_var;
  }
}

// This function is used by `InsertShadow` and `DeleteShadow`.
// When call this function, the lock is acquired, so we don't
// need to acquire the lock agagin. Otherwise it causes a deadlock.
ShadowVar* ShadowManager::GetShadowUnlock(string name){
  auto iter = shadows_.find(name);
  if(iter == shadows_.end()){
    return nullptr;
  }
  return iter->second;
}


ShadowVar* ShadowManager::GetShadow(string name){
  mutex_lock l(mu_);
  auto iter = shadows_.find(name);
  if(iter == shadows_.end()){
    return nullptr;
  }
  return iter->second;
}

// Get all shadows in this worker.
void ShadowManager::GetAllShadowNames(std::vector<string>* all_shadow_names,
                                      std::vector<int64>* all_shadow_steps){
  mutex_lock l(mu_);
  for(auto iter=shadows_.begin(); iter!=shadows_.end(); ++iter){
    all_shadow_names->emplace_back(iter->second->name());
    all_shadow_steps->emplace_back(iter->second->global_step());
  }
}

int ShadowManager::number_shadows(){
  return shadows_.size();
}

// The global shadow manager.
ShadowManager g_shadow_manager;

} // end namespace tensorflow