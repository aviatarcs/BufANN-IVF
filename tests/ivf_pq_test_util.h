// Minimal harness for the self-checking IVF-PQ tests: a TestCase prints its
// name, records failed checks, and reports PASS/FAIL from done().

#pragma once

#include <unistd.h>

#include <exception>
#include <iostream>
#include <string>

#include "ann_exception.h"

class TestCase {
public:
    explicit TestCase(const std::string& name) {
        std::cout << "[Test] " << name << "..." << std::endl;
    }

    bool check(bool cond, const std::string& msg) {
        if (!cond) {
            std::cout << "  FAIL: " << msg << std::endl;
            _pass = false;
        }
        return cond;
    }

    template<typename Fn>
    void expect_throw(const std::string& what, Fn&& fn) {
        try {
            fn();
            check(false, what + " did not throw");
        } catch (const diskann::ANNException&) {
        }
    }

    // For code paths that throw std::invalid_argument / std::runtime_error
    // rather than ANNException (the public API does).
    template<typename Fn>
    void expect_throw_any(const std::string& what, Fn&& fn) {
        try {
            fn();
            check(false, what + " did not throw");
        } catch (const std::exception&) {
        } catch (const diskann::ANNException&) {
        }
    }

    bool done() {
        std::cout << "  " << (_pass ? "PASS" : "FAIL") << std::endl;
        return _pass;
    }

private:
    bool _pass = true;
};

inline std::string temp_path(const std::string& stem) {
    return "/tmp/" + stem + "_" + std::to_string(static_cast<uint64_t>(getpid()));
}
