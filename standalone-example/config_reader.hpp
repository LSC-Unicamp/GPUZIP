#pragma once
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "common/GPUZIPConfig.h"

class ConfigReader {
public:
    explicit ConfigReader(const std::string& path) {
        std::ifstream f(path);
        if (!f.is_open())
            throw std::runtime_error("Cannot open config file: " + path);
        std::string line;
        while (std::getline(f, line)) {
            auto hash = line.find('#');
            if (hash != std::string::npos) line = line.substr(0, hash);
            auto eq = line.find('=');
            if (eq == std::string::npos) continue;
            std::string key = trim(line.substr(0, eq));
            std::string val = trim(line.substr(eq + 1));
            if (!key.empty() && !val.empty()) data_[key] = val;
        }
    }

    int    get_int   (const std::string& k, int    def = 0)    const { return has(k) ? std::stoi(data_.at(k))   : def; }
    double get_double(const std::string& k, double def = 0.0)  const { return has(k) ? std::stod(data_.at(k))   : def; }
    float  get_float (const std::string& k, float  def = 0.0f) const { return has(k) ? std::stof(data_.at(k))   : def; }
    bool   has       (const std::string& k) const { return data_.count(k) > 0; }

private:
    std::unordered_map<std::string, std::string> data_;

    static std::string trim(std::string s) {
        const char* ws = " \t\r\n";
        s.erase(0, s.find_first_not_of(ws));
        auto last = s.find_last_not_of(ws);
        if (last != std::string::npos) s.erase(last + 1);
        else s.clear();
        return s;
    }
};

inline gpuzip_config_t load_gpuzip_config(const ConfigReader& cfg) {
    gpuzip_config_t c{};
    c.checkpointing_algorithm     = cfg.get_int("checkpointing_algorithm", 1);
    c.cache_capacity              = cfg.get_int("cache_capacity", 4);
    c.compressor                  = cfg.get_int("compressor", 2);
    c.log_level                   = cfg.get_int("log_level", 1);
    c.enable_performance_log      = false;
    c.enable_compression_rate_log = false;
    c.compression_factor          = cfg.get_float("compression_factor", 0.0f);
    c.trace_file_path             = nullptr;
    c.revolve_log_level           = cfg.get_int("revolve_log_level", 0);
    c.zfp_bit_rate                = cfg.get_int("zfp_bit_rate", 8);
    c.cuszp_err_bound             = cfg.get_double("cuszp_err_bound", 1e-4);
    c.bitcomp_delta_config        = cfg.get_int("bitcomp_delta_config", 2);
    c.bitcomp_range_fraction      = cfg.get_double("bitcomp_range_fraction", 0.0);
    c.bitcomp_num_sigma           = cfg.get_double("bitcomp_num_sigma", 0.0);
    c.bitcomp_delta               = cfg.get_double("bitcomp_delta", 1e-8);
    c.bitcomp_algorithm           = cfg.get_int("bitcomp_algorithm", 0);
    return c;
}
