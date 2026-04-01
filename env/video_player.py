
import ipywidgets as widgets
from IPython.display import display, Video, clear_output
import os
import glob

class VideoSelector:
    """
    A Jupyter widget to select and display MP4 videos from a directory.
    """
    def __init__(self, default_directory='videos'):
        """
        Initializes the widget components and their logic.
        
        Args:
            default_directory (str): The default folder to look for videos in.
        """
        # --- 1. Create the Widgets (as instance attributes) ---
        self.directory_input = widgets.Text(
            value=default_directory,
            placeholder='Type a directory path',
            description='Directory:',
            style={'description_width': 'initial'}
        )
        self.update_button = widgets.Button(
            description='Refresh Videos',
            button_style='info',
            tooltip='Find MP4 files in the directory'
        )
        self.update_button.layout.width = '150px'
        self.video_dropdown = widgets.Dropdown(
            options=['--- Select a video ---'],
            description='Video File:',
            disabled=True,
            style={'description_width': 'initial'}
        )
        self.video_output = widgets.Output()

        # --- 2. Link the Logic to the Widgets ---
        self.update_button.on_click(self._on_button_clicked)
        self.video_dropdown.observe(self._on_dropdown_change, names='value')

        # --- 3. Assemble the Final Layout ---
        self.app_layout = widgets.VBox([
            self.directory_input, 
            self.update_button, 
            self.video_dropdown, 
            self.video_output
        ])

    def _on_button_clicked(self, b):
        """Finds MP4 files and updates the dropdown menu."""
        path = self.directory_input.value
        with self.video_output:
            clear_output()
            if not os.path.isdir(path):
                print(f"Error: Directory not found at '{path}'")
                return
            
        video_files = glob.glob(os.path.join(path, '*.mp4'))
        filenames = [os.path.basename(f) for f in video_files]
        filenames.sort()
        
        if not filenames:
            with self.video_output:
                print(f"No .mp4 files found in '{path}'")
            self.video_dropdown.options = ['--- No videos found ---']
            self.video_dropdown.disabled = True
        else:
            self.video_dropdown.options = ['--- Select a video ---'] + filenames
            self.video_dropdown.disabled = False

    def _on_dropdown_change(self, change):
        """Displays the selected video."""
        if change['type'] == 'change' and change['name'] == 'value':
            selected_file = change['new']
            with self.video_output:
                clear_output(wait=True)
                if selected_file and selected_file not in ['--- Select a video ---', '--- No videos found ---']:
                    video_path = os.path.join(self.directory_input.value, selected_file)
                    print(f"Displaying: {video_path}")
                    display(Video(video_path, embed=True, width=800))
    
    def _ipython_display_(self):
        """
        This special method tells Jupyter how to display this object.
        When you call display(instance_of_VideoSelector), this method is run.
        """
        display(self.app_layout)