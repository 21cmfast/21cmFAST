from py21cmmc._21cmfast import initial_conditions, perturb_field, UserParams, CosmoParams, ionize_box, FlagOptions, AstroParams, spin_temperature, brightness_temperature

z_final = 6.0
redshift = z_final*1.0001
z_step_factor = 1.02
z_HEAT_MAX = 35.0

while redshift < z_HEAT_MAX:
	prev_redshift = redshift
	redshift = (1+redshift)*z_step_factor - 1

redshift = prev_redshift
prev_redshift = z_HEAT_MAX

init_boxes = initial_conditions(UserParams(),CosmoParams(RANDOM_SEED=1),regenerate=True)

perturb_field_finalz = perturb_field(6.0,init_boxes)

initial_create_Ts = True

while redshift > z_final:	

	new_perturb_field = perturb_field(redshift,init_boxes)

	if initial_create_Ts is True:
		SpinTempBox = spin_temperature(redshift=redshift, perturbed_field=perturb_field_finalz, z_heat_max=z_HEAT_MAX,
			flag_options=FlagOptions(USE_TS_FLUCT=True,INHOMO_RECO=True))
		
		new_ionize_box = ionize_box(redshift=redshift,perturbed_field=new_perturb_field,
			do_spin_temp=True,spin_temp=SpinTempBox,flag_options=FlagOptions(USE_TS_FLUCT=True,INHOMO_RECO=True))

		initial_create_Ts = False		
	else:		
		SpinTempBox = spin_temperature(redshift=redshift, perturbed_field=perturb_field_finalz, z_heat_max=z_HEAT_MAX,
			previous_spin_temp=SpinTempBox,flag_options=FlagOptions(USE_TS_FLUCT=True,INHOMO_RECO=True))

		new_ionize_box = ionize_box(redshift=redshift,perturbed_field=new_perturb_field,previous_ionize_box=new_ionize_box,
			do_spin_temp=True,spin_temp=SpinTempBox,flag_options=FlagOptions(USE_TS_FLUCT=True,INHOMO_RECO=True))

	prev_redshift = redshift
	redshift = ((1+prev_redshift) / z_step_factor - 1)
