//New file to store functions that deal with the interpolation tables, since they are used very often.
//  We use regular grid tables since they are faster to evaluate (we always know which bin we are in)
//  So I'm making a general function for the 1D and 2D cases
//TODO: find out if the compiler optimizes this if it's called with the same x or y, (first 3 points are the same)
//  Because there are some points such as the frequency integral tables where we evaluate an entire column of one 2D table
//  Otherwise write a function which saves the left_edge and right_edge for faster evaluation

struct RGTable1D{
    int n_bin;
    double x_min;
    double x_width;

    double *y_arr;
};

struct RGTable2D{
    int nx_bin, ny_bin;
    double x_min, y_min;
    double x_width, y_width;

    double **z_arr;

    double saved_ll, saved_ul; //for future acceleration
};

double EvaluateRGTable1D(double x, double *y_arr, double x_min, double x_width){
    int idx = (int)floor((x - x_min)/x_width);
    double table_val = x_min + x_width*(double)idx;
    double interp_point = (x - table_val)/x_width;

    return y_arr[idx]*(1-interp_point) + y_arr[idx+1]*(interp_point);
}

//some tables are floats but I still need to return doubles
double EvaluateRGTable1D_f(double x, float *y_arr, float x_min, float x_width){
    int idx = (int)floor((x - x_min)/x_width);
    double table_val = x_min + x_width*(float)idx;
    double interp_point = (x - table_val)/x_width;

    return y_arr[idx]*(1-interp_point) + y_arr[idx+1]*(interp_point);
}

double EvaluateRGTable2D(double x, double y, double **z_arr, double x_min, double x_width, double y_min, double y_width){
    int x_idx = (int)floor((x - x_min)/x_width);
    int y_idx = (int)floor((y - y_min)/y_width);

    double x_table = x_min + x_width*(double)x_idx;
    double y_table = y_min + y_width*(double)y_idx;

    double interp_point_x = (x - x_table)/x_width;
    double interp_point_y = (y - y_table)/y_width;

    double left_edge, right_edge, result;

    // LOG_DEBUG("2D Interp: val (%.2e,%.2e) min (%.2e,%.2e) wid (%.2e,%.2e)",x,y,x_min,y_min,x_width,y_width);
    // LOG_DEBUG("2D Interp: idx (%d,%d) tbl (%.2e,%.2e) itp (%.2e,%.2e)",x_idx,y_idx,x_table,y_table,interp_point_x,interp_point_y);
    // LOG_DEBUG("2D Interp: table cornders (%.2e,%.2e,%.2e,%.2e)",z_arr[x_idx][y_idx],z_arr[x_idx][y_idx+1],z_arr[x_idx+1][y_idx],z_arr[x_idx+1][y_idx+1]);

    left_edge = z_arr[x_idx][y_idx]*(1-interp_point_y) + z_arr[x_idx][y_idx+1]*(interp_point_y);
    right_edge = z_arr[x_idx+1][y_idx]*(1-interp_point_y) + z_arr[x_idx+1][y_idx+1]*(interp_point_y);

    result = left_edge*(1-interp_point_x) + right_edge*(interp_point_x);
    // LOG_DEBUG("result %.6e",result);

    return result;
}