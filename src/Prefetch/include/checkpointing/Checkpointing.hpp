#pragma once

#include <cstdio>
#include <iostream>

/**
 * @enum ActionType
 * @brief Defines the types of checkpointing actions.
 * @author Thiago Maltempi <maltempi@ic.unicamp.br>
 * @author Sandro Rigo <srigo@ic.unicamp.br>
 * @date Dec 5, 2024
 * 
 * Actions:
 * - FORWARD: Advance computation without saving or restoring.
 * - SAVE: Save the current state as a checkpoint.
 * - RESTORE: Restore a previously saved checkpoint.
 * - BACKWARD: Perform a reverse computation.
 * - TERMINATE: Terminate the checkpointing process.
 * - ACTION_ERROR: Represents an invalid or unrecognized action.
 */
enum ActionType { ACTION_FORWARD, ACTION_SAVE, ACTION_RESTORE, ACTION_BACKWARD, ACTION_TERMINATE, ACTION_ERROR };

/**
 * @enum CheckpointingImplementation
 * @brief Enumerates checkpointing implementation strategies.
 * @author Thiago Maltempi <maltempi@ic.unicamp.br>
 * @author Sandro Rigo <srigo@ic.unicamp.br>
 * @date Dec 5, 2024
 * 
 * - TRACE: Follows a predefined trace for checkpointing actions.
 * - REVOLVE: Uses the Revolve algorithm for checkpointing.
 */
enum CheckpointingImplementation { TRACE, REVOLVE };

/**
 * @struct Action
 * @brief Represents a single checkpointing action.
 * @author Thiago Maltempi <maltempi@ic.unicamp.br>
 * @author Sandro Rigo <srigo@ic.unicamp.br>
 * @date Dec 5, 2024
 * 
 * Contains the following fields:
 * - start: The starting timestep for the action.
 * - end: The ending timestep for the action.
 * - actionType: The type of action to be performed (see `ActionType`).
 */
struct Action {
  int ts;
  ActionType actionType;
  Action(int ts, ActionType a) : ts(ts), actionType(a) {}
};

/**
 * @class Checkpointing
 * @brief Abstract base class for implementing checkpointing mechanisms.
 * @author Thiago Maltempi <maltempi@ic.unicamp.br>
 * @author Sandro Rigo <srigo@ic.unicamp.br>
 * @date Dec 5th, 2024
 *
 * Provides a common interface for derived checkpointing implementations such
 * as `RevolveCheckpointing` and `TraceCheckpointing`. Tracks computational
 * steps, checkpoints, and iteration indices.
 */
class Checkpointing {

protected:
  int it = 0;
  int steps = 0;
  int info = 1;
  int snaps = 0;

  /**
   * @brief Resets the checkpointing state (to be implemented by derived
   * classes).
   */
  virtual void reset() = 0;

  /**
   * @brief Retrieves the next action in the checkpointing process.
   *
   * @return The next `Action` object, detailing the type and range of the
   * action.
   */
  virtual Action getAction() = 0;

  /**
   * @brief Calculates the total number of checkpoints required.
   *
   * @return The total number of checkpoints.
   */
  virtual int getNumberOfCheckpoints() = 0;

public:
  /**
   * @brief Constructs a `Checkpointing` object.
   *
   * @param _steps The total number of computational steps for checkpointing.
   */
  Checkpointing(int _steps) { steps = _steps; }

  /**
   * @brief Retrieves the total number of checkpoints.
   *
   * @return The number of checkpoints.
   */
  int GetNumberOfCheckpoints() { return getNumberOfCheckpoints(); }

  /**
   * @brief Retrieves the next checkpointing action and increments the iteration
   * index.
   *
   * @return The next `Action` object.
   */
  Action GetAction() {
    it++;
    return getAction();
  }

  /**
   * @brief Resets the checkpointing process to its initial state.
   */
  void Reset() {
    it = -1;
    reset();
  }

  /**
   * @brief Gets the current iteration index.
   *
   * @return The current iteration index.
   */
  int GetIt() { return it; }

  /**
   * @brief Gets the total number of computational steps.
   *
   * @return The total number of steps.
   */
  int GetSteps() { return steps; }
};
