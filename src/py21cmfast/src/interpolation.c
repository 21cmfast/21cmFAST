//New file to store functions that deal with the interpolation tables, since they are used very often.
//  We use regular grid tables since they are faster to evaluate (we always know which bin we are in)
//  So I'm making a general function for the 1D and 2D cases
//TODO: find out if the compiler optimizes this if it's called with the same x or y, (first 3 points are the same)
//  Because there are some points such as the frequency integral tables where we evaluate an entire column of one 2D table
//  Otherwise write a function which saves the left_edge and right_edge for faster evaluation
//TODO: make a print_interp_error(array,x,y) function which prints the cell corners, interpolation points etc

static struct UserParams * user_params_it;

void Broadcast_struct_global_IT(struct UserParams *user_params){
    user_params_it = user_params;
}

//TODO: for the moment the tables are still in global arrays, but will move to these structs soon
//TODO: sort out if we actually need single precision tables
struct RGTable1D{
    int n_bin;
    double x_min;
    double x_width;

    double *y_arr;
    int allocated;
};

struct RGTable2D{
    int nx_bin, ny_bin;
    double x_min, y_min;
    double x_width, y_width;

    double **z_arr;

    double saved_ll, saved_ul; //for future acceleration
    int allocated;
};

struct RGTable1D_f{
    int n_bin;
    double x_min;
    double x_width;

    float *y_arr;
    int allocated;
};

struct RGTable2D_f{
    int nx_bin, ny_bin;
    double x_min, y_min;
    double x_width, y_width;

    float **z_arr;

    double saved_ll, saved_ul; //for future acceleration
    int allocated;
};

void allocate_RGTable1D(int n_bin, struct RGTable1D * ptr){
    ptr->n_bin = n_bin;
    ptr->y_arr = calloc(n_bin,sizeof(double));
    ptr->allocated = true;
}

void allocate_RGTable1D_f(int n_bin, struct RGTable1D_f * ptr){
    ptr->n_bin = n_bin;
    ptr->y_arr = calloc(n_bin,sizeof(float));
    ptr->allocated = true;
}

void free_RGTable1D(struct RGTable1D * ptr){
    if(ptr->allocated){
        free(ptr->y_arr);
        ptr->allocated = false;
    }
}

void free_RGTable1D_f(struct RGTable1D_f * ptr){
    if(ptr->allocated){
        free(ptr->y_arr);
        ptr->allocated = false;
    }
}

void allocate_RGTable2D(int n_x, int n_y, struct RGTable2D * ptr){
    int i;
    ptr->nx_bin = n_x;
    ptr->ny_bin = n_y;

    ptr->z_arr = calloc(n_x,sizeof(double*));
    for(i=0;i<n_x;i++){
        ptr->z_arr[i] = calloc(n_y,sizeof(double));
    }
    ptr->allocated = true;
}

void allocate_RGTable2D_f(int n_x, int n_y, struct RGTable2D_f * ptr){
    int i;
    ptr->nx_bin = n_x;
    ptr->ny_bin = n_y;

    ptr->z_arr = calloc(n_x,sizeof(float*));
    for(i=0;i<n_x;i++){
        ptr->z_arr[i] = calloc(n_y,sizeof(float));
    }
    ptr->allocated = true;
}

void free_RGTable2D_f(struct RGTable2D_f * ptr){
    int i;
    if(ptr->allocated){
        for(i=0;i<ptr->nx_bin;i++)
            free(ptr->z_arr[i]);
        free(ptr->z_arr);
        ptr->allocated = false;
    }
}

void free_RGTable2D(struct RGTable2D * ptr){
    int i;
    if(ptr->allocated){
        for(i=0;i<ptr->nx_bin;i++)
            free(ptr->z_arr[i]);
        free(ptr->z_arr);
        ptr->allocated = false;
    }
}

double EvaluateRGTable1D(double x, struct RGTable1D *table){
    double x_min = table->x_min;
    double x_width = table->x_width;
    int idx = (int)floor((x - x_min)/x_width);
    double table_val = x_min + x_width*(double)idx;
    double interp_point = (x - table_val)/x_width;
    // LOG_DEBUG("1D: x %.6e (min %.2e wid %.2e) -> idx %d -> tbl %.6e -> itp %.6e",x, x_min, x_width,idx,table_val,interp_point);

    //a + f(a-b) is one fewer operation but less precise
    double result = table->y_arr[idx]*(1-interp_point) + table->y_arr[idx+1]*(interp_point);

    // LOG_DEBUG("-> result %.2e",result);

    return result;
}

double EvaluateRGTable2D(double x, double y, struct RGTable2D *table){
    double x_min = table->x_min;
    double x_width = table->x_width;
    double y_min = table->y_min;
    double y_width = table->y_width;
    int x_idx = (int)floor((x - x_min)/x_width);
    int y_idx = (int)floor((y - y_min)/y_width);

    double x_table = x_min + x_width*(double)x_idx;
    double y_table = y_min + y_width*(double)y_idx;

    double interp_point_x = (x - x_table)/x_width;
    double interp_point_y = (y - y_table)/y_width;

    double left_edge, right_edge, result;

    // LOG_DEBUG("2D Interp: val (%.2e,%.2e) min (%.2e,%.2e) wid (%.2e,%.2e)",x,y,x_min,y_min,x_width,y_width);
    // LOG_DEBUG("2D Interp: idx (%d,%d) tbl (%.2e,%.2e) itp (%.2e,%.2e)",x_idx,y_idx,x_table,y_table,interp_point_x,interp_point_y);
    // LOG_DEBUG("2D Interp: table corners (%.2e,%.2e,%.2e,%.2e)",table->z_arr[x_idx][y_idx],table->z_arr[x_idx][y_idx+1],table->z_arr[x_idx+1][y_idx],table->z_arr[x_idx+1][y_idx+1]);

    left_edge = table->z_arr[x_idx][y_idx]*(1-interp_point_y) + table->z_arr[x_idx][y_idx+1]*(interp_point_y);
    right_edge = table->z_arr[x_idx+1][y_idx]*(1-interp_point_y) + table->z_arr[x_idx+1][y_idx+1]*(interp_point_y);

    result = left_edge*(1-interp_point_x) + right_edge*(interp_point_x);
    // LOG_DEBUG("result %.6e",result);

    return result;
}

//some tables are floats but I still need to return doubles
double EvaluateRGTable1D_f(double x, struct RGTable1D_f *table){
    double x_min = table->x_min;
    double x_width = table->x_width;
    int idx = (int)floor((x - x_min)/x_width);
    double table_val = x_min + x_width*(float)idx;
    double interp_point = (x - table_val)/x_width;

    return table->y_arr[idx]*(1-interp_point) + table->y_arr[idx+1]*(interp_point);
}

double EvaluateRGTable2D_f(double x, double y, struct RGTable2D_f *table){
    double x_min = table->x_min;
    double x_width = table->x_width;
    double y_min = table->y_min;
    double y_width = table->y_width;
    int x_idx = (int)floor((x - x_min)/x_width);
    int y_idx = (int)floor((y - y_min)/y_width);

    double x_table = x_min + x_width*(double)x_idx;
    double y_table = y_min + y_width*(double)y_idx;

    double interp_point_x = (x - x_table)/x_width;
    double interp_point_y = (y - y_table)/y_width;

    double left_edge, right_edge, result;

    // LOG_DEBUG("2D Interp: val (%.2e,%.2e) min (%.2e,%.2e) wid (%.2e,%.2e)",x,y,x_min,y_min,x_width,y_width);
    // LOG_DEBUG("2D Interp: idx (%d,%d) tbl (%.2e,%.2e) itp (%.2e,%.2e)",x_idx,y_idx,x_table,y_table,interp_point_x,interp_point_y);
    // LOG_DEBUG("2D Interp: table corners (%.2e,%.2e,%.2e,%.2e)",z_arr[x_idx][y_idx],z_arr[x_idx][y_idx+1],z_arr[x_idx+1][y_idx],z_arr[x_idx+1][y_idx+1]);

    left_edge = table->z_arr[x_idx][y_idx]*(1-interp_point_y) + table->z_arr[x_idx][y_idx+1]*(interp_point_y);
    right_edge = table->z_arr[x_idx+1][y_idx]*(1-interp_point_y) + table->z_arr[x_idx+1][y_idx+1]*(interp_point_y);

    result = left_edge*(1-interp_point_x) + right_edge*(interp_point_x);
    // LOG_DEBUG("result %.6e",result);

    return result;
}

