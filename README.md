# jigsaw-puzzle-solver
Software developed by a group of students from The University of Melbourne for autonomously solving jigsaw puzzles.

To use this library you will need to:
 - install the following packages:
   - opencv
   - numpy
   - imageio
 - install the following packages (optional):
   - git
   - ipython-autotime
   - notebook
 - clone repository to your local machine.
   - !git clone -l -s https://USERNAME:PASSWORD@github.com/BenSoltau/jigsaw-puzzle-solver.git
 - Follow the instructions in the Jupyter Notebook example.ipynb to run the program.

## File Tree: ##
 - example.ipynb
 - puzzle_solver.py
   - extractor.py
     - aligner.py
     - bgr_data.py
     - center_finder.py
     - clearance_radius.py
     - colour_identification.py
     - contour_finder.py
     - convexity.py
     - corner_finder.py
     - extractor.py
     - hull_creator.py
     - image_import.py
     - locks.py
     - piece_types.py
     - processed_data.py
     - side_separator.py

   - solver.py
     - assistant.py
     - backtracker.py
     - comparator.py
     - template.py
     - updator.py
     - utils.py
   - target_solver.py
     - cropping.py
 - global_settings.py
 - graphics.py
 - utils.py
 - datasets
