# GUI Implementation 

To complement the probabilistic Monte Carlo simulations, we developed a Python-based, interactive GUI to deterministically visualize the evolution of mechanical feasibility throughout the stair ascent cycle. This tool integrates posture-dependent moment arms and muscle force capacities to compute the Feasible Torque Set (FTS) in 3D joint space and projects it into the Feasible Force Set (FFS) using the pseudo-inverted, transposed, kinematic Jacobian. By calculating the Minkowski sum of active muscle generators, the GUI defines the absolute theoretical limits of force production for a specific subject model at every frame of the ascent cycle. This platform served as a "virtual laboratory" to test the robustness of the limb by selectively deactivating muscles to simulate severe focal sarcopenia, directly addressing the research question of identifying the most critical muscles for stair ascent. 

![GUI logical flowchart](https://github.com/ek-usc/FrailtyProject-StairClimbingModelGUI/blob/master/figures/GUI_flowchart.png)

GUI Logical Flowchart: Details inputs/outputs and logical process of the Python code to produce FTS/FFS for each frame in ascent cycle. 


### Interpretation of GUI Results

The deterministic results from the GUI strongly validated the sensitivity analysis observed in the Monte Carlo simulations. The visualization revealed that the mechanical redundancy of the limb is highly anisotropic--while the limb retains significant capacity in horizontal directions after muscle loss, the vertical force capacity is critically fragile. Specifically, during push-off at frame 12, toggling off the soleus muscle alone reduced the maximum vertical component of the FFS by ~56.3% (from -2700 N to -1180 N). The simultaneous removal of the plantar flexor group (medial and lateral gastrocnemius, soleus, tibialis posterior, flexor digitorum, and flexor hallucis) collapsed the feasible set, reducing vertical force capacity by ~89.4% (to -285 N). In both of these states, the required Ground Reaction Force (GRF) vector fell completely outside the feasible polygon, indicating an inability to support body weight and generate the necessary propulsion at that frame in the stair ascent cycle. These findings underscore that frailty in stair ascent is not a generalized weakness of all muscles in the lower limb, but is mostly mechanically characterized by the incapacity of the plantar flexors to meet the task’s GRF requirements during the stance and push-off phases. 

![GUI polytope comparison](https://github.com/ek-usc/Frailty_Project-Feasible_Torque_and_Force_Sets_Neuromechanics_GUI/blob/master/figures/polytope_comparisons.png)

FTS/FFS Comparison: Displays FTS/FFS with all muscles included (left), only soleus excluded (center), only medial gastrocnemius excluded (right); drastic shrinkage supports Monte Carlo sensitivity analysis—soleus and other plantar flexor muscles are most critical 


![GUI preview](https://github.com/ek-usc/FrailtyProject-StairClimbingModelGUI/blob/master/figures/frailtyGUI.png)

Preview of GUI

# Acknowledgements:
Gemini 3 Pro was used to provide a workflow framework and assist with coding.  

Initial prompt: "[We are] trying to model the human lower limb to study the diminishing feasible output forces and parameters as a person grows elderly, framing the problem in the context of what [we] learned in class. . . about torque, force, and muscle activation spaces. Also attached is a pdf compiling [our] research notes, resources, and tasks. Describe thoroughly, the GUI producer's role, and outline their workflow steps: "The other to write a script to visualize the output-figures to a video (for example, make a GUI, Users can choose muscles they interested in, then by dragging the progress bar, users can choose each leg posture, and corresponding feasible force set will appear next to the posture)." Make sure to also describe the required input parameters that the script will accept and read, before producing the user-interactive GUI. Also consider some useful, extra features for the GUI." 
