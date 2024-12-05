#pragma once

#include <stdio.h>
#include <stdlib.h>

#include <map>
#include <string>
#include <vector>

#include <fstream>
#include <iostream>
#include <sstream>

#include "../../common/GPUZIPLogger.cpp"
#include "../Checkpointing.hpp"

/**
 * @class TraceCheckpointing
 * @brief Implements a trace-based checkpointing mechanism.
 * @author Thiago Maltempi <maltempi@ic.unicamp.br>
 * @author Sandro Rigo <srigo@ic.unicamp.br>
 * @date Dec 5th, 2024
 *
 * This class reads a trace file to determine checkpointing actions, such as
 * SAVE, RESTORE, and FORWARD, enabling precise replay of checkpoint sequences.
 */
class TraceCheckpointing : public Checkpointing {

private:
  std::vector<Action> trace;
  int intermediaries = 0;

  /**
   * @brief Maps a string-based action to an `ActionType`.
   *
   * @param actionStr The string representation of an action (e.g., "SAVE").
   * @return The corresponding `ActionType` enum or `ACTION_ERROR` if invalid.
   */
  ActionType getActionType(const std::string &actionStr) {
    static std::map<std::string, ActionType> actionMap = {
        {"FORWARD", ACTION_FORWARD},        {"SAVE", ACTION_SAVE},
        {"SAVE_INTERMEDIARY", ACTION_SAVE}, {"RESTORE", ACTION_RESTORE},
        {"BACKWARD", ACTION_BACKWARD},      {"FIRSTURN", ACTION_BACKWARD},
        {"TERMINATE", ACTION_TERMINATE}};

    auto it = actionMap.find(actionStr);
    if (it != actionMap.end()) {
      return it->second;
    } else {
      return ACTION_ERROR;
    }
  }

  /**
   * @brief Loads the trace file and populates the `trace` vector.
   *
   * @param filename The name of the file containing the checkpointing trace.
   *
   * Reads the trace file line by line, interpreting actions and their
   * associated timesteps. Tracks the number of intermediary saves.
   */
  void loadTrace(const std::string &filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
      fprintf(stderr, "Error on opening file: %s\n", filename.c_str());
      exit(1);
    }

    std::string line;
    GPUZIPLogger::Debug("TRACE");

    int currIntermediaries = 0;

    while (std::getline(file, line)) {
      std::istringstream iss(line);
      int timestep;
      std::string actionStr;
      if (iss >> timestep >> actionStr) {
        GPUZIPLogger::Debug("%i %s\n", timestep, actionStr.c_str());

        if (currIntermediaries > 0 &&
            (actionStr == "RESTORE" || actionStr == "SAVE" ||
             actionStr == "BACKWARD")) {
          if (currIntermediaries > intermediaries) {
            intermediaries = currIntermediaries;
          }

          currIntermediaries = 0;
        }

        if (actionStr == "SAVE_INTERMEDIARY") {
          currIntermediaries++;
        }

        trace.push_back(Action(timestep, getActionType(actionStr)));
      }
    }

    GPUZIPLogger::Debug("ENDTRACE\n\n\n\n");

    file.close();
  }

protected:
  /**
   * @brief Resets the internal state of the checkpointing process.
   */
  void reset() override {}

  /**
   * @brief Retrieves the next action from the trace.
   *
   * @return The next `Action` object, or a `TERMINATE` action if the end of the
   * trace is reached.
   */
  Action getAction() override {
    if (it < 0 || it >= static_cast<int>(trace.size())) {
      return Action(-1, ACTION_TERMINATE);
    }

    return trace[it];
  }

  /**
   * @brief Calculates the total number of checkpoints in the trace.
   *
   * @return The number of checkpoints, including intermediary saves.
   */
  int getNumberOfCheckpoints() override {
    if (snaps > 0) {
      return snaps;
    }

    // Count the number of SAVE actions in the selected dataset
    for (const auto &entry : trace) {
      if (entry.actionType == ACTION_RESTORE) {
        break;
      }
      if (entry.actionType == ACTION_SAVE) {
        snaps++;
      }
    }

    snaps = snaps + intermediaries;

    return snaps;
  }

public:
  /**
   * @brief Constructor for the TraceCheckpointing class.
   *
   * @param steps The number of computational steps for the checkpointing
   * process.
   * @param filename The name of the file containing the checkpointing trace.
   *
   * Initializes the base `Checkpointing` class and loads the trace from the
   * specified file.
   */
  TraceCheckpointing(int steps, std::string filename) : Checkpointing(steps) {
    loadTrace(filename.c_str());
  }
};
