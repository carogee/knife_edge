def knife_edge(self,det,motor,start,stop,steps,n,guess): #n=#of measurements at each step                              
        from databroker import Broker
	import numpy as np
        from scipy.optimize import minimize
        import matplotlib.pyplot as plt
        from bluesky.preprocessors import run_decorator
	from ophyd.signal import EpicsSignal
        from bluesky.plan_stubs import abs_set
        from bluesky.callbacks.best_effort import BestEffortCallback
	from scipy import special
        from scipy.optimize import curve_fit
        from scipy.special import erfc

        print("Doing knife edge laser scan on motor:", motor)
        ljh_jet_x=EpicsSignal('XCS:LJH:JET:X')
        labmax=EpicsSignal('XCS:LPW:01:DATA_PRI')

        #RE(scan([labmax], motor, start, stop, steps)) #, per_step=num_measurements)                                       
        positions = np.repeat(np.linspace(start,stop,steps),n)
        #positions = [1,2,3,4,5]                                                                                           
        print("positions",positions)

        RE = RunEngine()
        db = Broker.named('temp')
        RE.subscribe(db.insert)

	det_arr = []
	
        def labmax_range_scan(detector, motor, positions):
            for pos in positions:
		print("pos", pos)
                # Move motor to the next position                                                                          
                yield from abs_set(ljh_jet_x, pos, wait=True)

                # Trigger the detector and read its value                                                                  
                yield from count([detector])
                det_value = list(detector.read().values())[0]['value']

                # Apply conditional logic based on detector value using epics.caput                                        
                if det_value < 3e-06:
                    # Set detector to setting A using caput (e.g., change some configuration)                              
                    print("det_value", det_value)
                    os.system('caput XCS:LPW:01:SETRANGE 3e-06')
                else:
                    # Set detector to setting B using caput (e.g., change some configuration)                              
                    # Set labmax range                                                                                    \
                                                                                                                           
                    print("det_value", det_value)
                    os.system('caput XCS:LPW:01:SETRANGE 3e-05')

        

                # Read the detector again after setting change                                                             
                print("reading detector again after setting change")
                yield from count([detector],num=1,delay=1)
                det_value = list(detector.read().values())[0]['value']
                print("det valued read", det_value)
                det_arr.append(det_value)
                print("appended values")
            yield from list_scan([labmax], motor, positions)
            print("closing run")


        print("starting the plan")
        uids=RE(labmax_range_scan(labmax, motor, positions)) #motor                                                        
        #print("plan completed. UID:", uid)                                                                                
        #except Exception as e:                                                                                            
        #    print(f"Error occurred: {e}")                                                                                 



        header = db[uids]
        h = db[-1]
        t = h.table()

        print("header.table",t)
        print('ljh_jet_x',t['ljh_jet_x'])
        print('det',t['XCS:LPW:01:DATA_PRI'])

	power = det_arr
        position = t['ljh_jet_x']
	
        #Fitting
        def cauchy_loss(params, x, y, c):
            A, mu, sigma = params
            y_pred = A * np.exp(-(x - mu)**2 / (2 * sigma**2))
            residual = y - y_pred
            return np.log(1 + (residual / c)**2).sum()

        # Function to define the Gaussian function for curve fitting                                                       
        def gauss(params, x):
            A, mu, sigma = params
            return A * np.exp(-(x - mu)**2 / (2 * sigma**2))

        # Objective function for optimization                                                                              
        def objective(params, x, y, c):
            return cauchy_loss(params, x, y, c)

	#Error function                                                                                                                                                          
        def error_function(x, a, b, c, d):
            #a,b,c,d = params                                                                                                                                                    
            return a * special.erfc(b * (x - c)) + d


        # Step 1: Create a dictionary to collect y values for each x                                                                                                             
        xy_dict = {}

        # Step 2: Populate the dictionary                                                                                                                                        
        for x, y in zip(position, power):
            if x not in xy_dict:
                xy_dict[x] = []
            xy_dict[x].append(y)

        # Step 3: Compute the averages                                                                                                                                           
        x_unique = []
        y_avg = []

        for x in sorted(xy_dict.keys()):  # Sort keys to maintain order                                                                                                          
            x_unique.append(x)
            y_avg.append(np.mean(xy_dict[x]))

	# Convert lists to numpy arrays (optional)                                                                                                                               
        pos_unique = np.array(x_unique)

        int_avg = np.array(y_avg)

        # Print the results                                                                                                                                                      
        print("Unique x values:", pos_unique)
        print("Averaged y values:", int_avg)

	#save the data in a file
	combined_array = np.column_stack((pos_unique,int_avg))

	folder_path = '/cds/group/xcs/knife_edge'
	file_path = os.path.join(folder_path, 'my_data.dat')
	np.savetxt(file_path, combined_array, delimiter=' ', fmt='%.4f')

        # Fit the data to the error function model                                                                                                                               
        initial_guess_erf = [(min(int_avg)-max(int_avg))/2, 2/(min(int_avg)-max(int_avg)), np.mean(pos_unique), max(int_avg)]
        params_erf, covariance = curve_fit(error_function, pos_unique, int_avg, p0=initial_guess_erf)
        print("initial guess", initial_guess_erf)

        # Plot the original data and the fitted curve                                                                                                                            
        plt.scatter(pos_unique, int_avg, label='Data')
        plt.plot(position, error_function(position, *params_erf), color='red', label='Fitted curve')
        plt.legend()
        plt.xlabel('position')
        plt.ylabel('power')
        plt.title('Fitting Data with Error Function')
        plt.show()

        print("Fitted parameters (a, b, c, d):", params_erf)


        FWHM_erf = (2*np.sqrt(2*np.log(2)))/params_erf[1]
        print(2*np.sqrt(2*np.log(2)))
        print("b", params_erf[1])
        print("Erf FWHM", FWHM_erf)

        # Initial guess for Gaussian parameters                                                                                     
        initial_guess = [0.01, np.mean(t['ljh_jet_x']), 0.001*guess]

        print("initial guess", initial_guess)

        

        # Perform optimization                                                                                             
        #result = minimize(objective, initial_guess, args=(power_arr, pos_arr, 1))                                         
        result = minimize(objective, initial_guess, args=(t['ljh_jet_x'], t['XCS:LPW:01:DATA_PRI'], 1))

        # Extract optimized parameters                                                                                     
        popt = result.x
        #print("popt", popt)                                                                                               

        # Calculating the full width at half maximum (FWHM)                                                                
        FWHM = 2 * np.sqrt(2 * np.log(2)) * popt[2]
        print("The FWHM is:", FWHM*1000, "um")

        # Plotting the data and the fit                                                                                    
        fig2 = plt.figure("Knife Edge Scan")
        plt.plot(t['ljh_jet_x'], t['XCS:LPW:01:DATA_PRI'], 'b.', label='Measurement Data')
        plt.plot(t['ljh_jet_x'], gauss(popt, t['ljh_jet_x']), 'r-', label='Gaussian Fit')
        plt.xlabel('Position [mm]')
        plt.ylabel('Intensity [J]')
	      plt.title('Edge Profile with Gaussian Fit (robust, Cauchy loss)')
        plt.legend()
        plt.show()
