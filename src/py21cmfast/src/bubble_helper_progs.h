/* Function prototypes for bubble_helper_progs.c */
#ifndef _BUBBLEHELP_H
#define _BUBBLEHELP_H

#ifdef __cplusplus
extern "C" {
#endif
//NOTE: This file is only used for the old bubble finding algorithm which updates the whole sphere
void update_in_sphere(float * box, int dimensions, int dimensions_ncf, float R, float xf, float yf, float zf);

#ifdef __cplusplus
}
#endif
#endif
