#pragma once

#include <stdio.h>
#include <stdlib.h>

#include <map>
#include <string>
#include <vector>

#include <fstream>
#include <iostream>
#include <sstream>

#include "revolve.c"

#include "../Checkpointing.hpp"

#include "../../common/GPUZIPLogger.cpp"

/**
 * @class RevolveCheckpointing
 * @brief Implements a checkpointing mechanism using the Revolve algorithm.
 * @author Thiago Maltempi <maltempi@ic.unicamp.br>
 * @author Sandro Rigo <srigo@ic.unicamp.br>
 * @date Dec 5th, 2024
 *
 * This class extends the base `Checkpointing` class to provide specific
 * checkpointing actions (save, restore, forward, backward, terminate) based on
 * the Revolve algorithm.
 */
class RevolveCheckpointing : public Checkpointing {

private:
  std::vector<Action>
      trace; ///< Stores the sequence of actions taken during checkpointing.
  int intermediaries = 0; ///< Number of intermediate steps between checkpoints.
  int revolveInfo = 0;    ///< Stores metadata related to the Revolve algorithm.
  int check = -1;         ///< Index of the last checkpoint taken.
  int fine = -1;          ///< The last iteration where a checkpoint is taken.
  int capo = 0;      ///< Stores the position of the last restored checkpoint.
  int itForward = 0; ///< Control Variable. Number of forward iterations since
                     ///< the last checkpoint.
  int forwardTarget =
      0; ///< Control Variable. Target iteration for the next forward step.

protected:
  /**
   * @brief Resets the internal state of the checkpointing process.
   *
   * Sets `check`, `fine`, and `capo` to their initial values.
   * This is typically called to reinitialize the checkpointing algorithm.
   */
  void reset() override {
    check = -1;
    capo = 0;
    itForward = 0;
    forwardTarget = 0;
    fine = steps;
  }

  /**
   * @brief Retrieves the last iteration where a checkpoint was taken.
   *
   * If `fine` has not been initialized, it is set to `steps` before returning.
   *
   * @return A pointer to the `fine` variable.
   */
  int *GetFine() {
    if (fine == -1) {
      fine = steps;
    }
    return &fine;
  }

  /**
   * @brief Retrieves the index of the last checkpoint taken.
   *
   * @return A pointer to the `check` variable.
   */
  int *GetCheck() { return &check; }

  /**
   * @brief Retrieves the position of the last restored checkpoint.
   *
   * @return A pointer to the `capo` variable.
   */
  int *GetCapo() { return &capo; }

  /**
   * @brief Determines the next action to perform in the checkpointing process.
   *
   * @return An `Action` object describing the next step, including its type and
   * relevant parameters.
   */
  Action getAction() override {
    if (forwardTarget > 0) {
      itForward++;

      bool last_it = itForward >= forwardTarget;
      int cur_capo = itForward;

      if (last_it) {
        itForward = 0;
        forwardTarget = 0;
      }

      return Action(cur_capo, ACTION_FORWARD);

    } else {
      auto revolveAction = revolve(GetCheck(), GetCapo(), GetFine(),
                                   GetNumberOfCheckpoints(), &revolveInfo);

      if (revolveAction == revolve_takeshot) {
        itForward = capo + 1;
        return Action(capo, ACTION_SAVE);
      } else if (revolveAction == revolve_restore) {
        itForward = capo + 1;
        return Action(capo, ACTION_RESTORE);
      } else if (revolveAction == revolve_advance) {
        if (itForward >= capo) {
          itForward = 0;
          forwardTarget = 0;
          return Action(capo, ACTION_FORWARD);
        } else {
          forwardTarget = capo;
          return Action(itForward, ACTION_FORWARD);
        }
      } else if (revolveAction == revolve_firsturn ||
                 revolveAction == revolve_youturn) {
        forwardTarget = 0;
        itForward = 0;
        return Action(capo, ACTION_BACKWARD);
      } else if (revolveAction == revolve_terminate) {
        forwardTarget = 0;
        itForward = 0;
        return Action(capo, ACTION_TERMINATE);
      } else {
        forwardTarget = 0;
        itForward = 0;
        GPUZIPLogger::Error("Irregular termination of Revolve: [CODE=%i]\n",
                            revolveAction);
        switch (revolveAction) {
        case 10:
          GPUZIPLogger::Error(
              "The number of checkpoints stored exceeds checkup. "
              "Recomendation: Increase constant 'checkup' and recompile.\n");
          break;
        case 11:
          GPUZIPLogger::Error("The number of checkpoints stored = %d exceeds "
                              "snaps = %d. Recomendation: Ensure 'snaps' > 0 "
                              "and increase initial 'fine'.\n",
                              check + 1, GetNumberOfCheckpoints());
          break;
        case 12:
          GPUZIPLogger::Error("Error occurs in numforw.\n");
          break;
        case 13:
          GPUZIPLogger::Error("Enhancement of 'fine', 'snaps' checkpoints "
                              "stored. Recomendation: Increase 'snaps'.\n");
          break;
        case 14:
          GPUZIPLogger::Error(
              "The number of snaps exceeds snapsup. Recomendation: Increase "
              "constant 'snapsup' and recompile.\n");
          break;
        case 15:
          GPUZIPLogger::Error(
              "The number of reps exceeds repsup. Recomendation: increase "
              "constant 'repsup' and recompile.\n");
        }

        return Action(capo, ACTION_ERROR);
      }
    }
  }

  /**
   * @brief Calculates the number of checkpoints required for the given steps.
   *
   * @return The number of checkpoints as determined by an adjustment algorithm.
   */
  int getNumberOfCheckpoints() override {
    if (snaps == 0) {
      snaps = adjust(steps);
    }
    return snaps;
  }

public:
  /**
   * @brief Constructor for the RevolveCheckpointing class.
   *
   * @param steps The number of computational steps for which checkpointing is
   * required.
   * @param _revolveInfo Additional information to initialize the Revolve
   * algorithm.
   *
   * Initializes the base `Checkpointing` class and sets up the internal state.
   */
  RevolveCheckpointing(int steps, int _revolveInfo)
      : Checkpointing(steps), revolveInfo(_revolveInfo), fine(steps) {}
};
