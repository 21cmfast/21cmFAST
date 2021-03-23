#include "cexcept.h"
define_exception_type(int);
extern struct exception_context the_exception_context[1];

struct exception_context the_exception_context[1];

// Our own error codes
#define SUCCESS 0
#define IOError 1
#define GSLError 2
#define ValueError 3
#define PhotonConsError 4
#define TableGenerationError 5
#define TableEvaluationError 6
#define InfinityorNaNError 7
#define MassDepZetaError 8
#define MemoryAllocError 9

#define GSL_ERROR(status) if(status>0) {LOG_ERROR("GSL Error Encountered (Code = %d): %s", status, gsl_strerror(status)); Throw(GSLError);}
