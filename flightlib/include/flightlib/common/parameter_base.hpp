#pragma once

#include <nlohmann/json.hpp>
using json = nlohmann::json;

namespace flightlib {

struct ParameterBase {
  ParameterBase();
  ParameterBase(const std::string& cfg_path);
  ParameterBase(const json& cfg_node);

  virtual ~ParameterBase();

  virtual bool valid() = 0;
  virtual bool loadParam() = 0;

  json cfg_node_;
};

}  // namespace flightlib
