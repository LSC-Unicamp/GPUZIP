#ifndef _REVOLVE_H_
#define _REVOLVE_H_

enum revolve_action { revolve_advance, revolve_takeshot, revolve_restore, revolve_firsturn, revolve_youturn, revolve_terminate, revolve_error };

#ifdef __cplusplus
extern "C" {
#endif
int maxrange(int ss, int tt);
int adjust(int steps);
int adjustsize(int *steps, int *snaps, int *reps);
enum revolve_action revolve(int *check, int *capo, int *fine, int snaps, int *info);
unsigned max_revolve_checkpoints();

#ifdef __cplusplus
}
#endif

#endif