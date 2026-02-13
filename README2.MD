# Video Review Application – Developer Documentation
This document provides a comprehensive overview of the Video Review Tool, a single-file Flask web application that allows users to review video footage, move files between folders, and revert changes. It is intended for developers who want to understand, maintain, or extend the application.

## 1. Introduction
The Video Review Tool is a lightweight, self-contained web server that:

* Serves video files from two directories: input_vid (videos awaiting review) and reviewed_vid (videos already reviewed).

* Provides a clean Bootstrap-based UI with embedded <video> players.

* Allows users to move a video from input_vid to reviewed_vid (mark as reviewed) or revert it back.

* Implements HTTP Range requests so videos are seekable and streaming works like a local file.

* Includes a retry mechanism and Windows-specific file-sharing flags to avoid “file in use” errors when moving files that are still being streamed.

The entire application is contained in one Python file – no separate template files or complex configuration.

# 2. Requirements & Dependencies
* Python 3.6 or newer

* Flask – micro web framework

All other modules used (os, shutil, mimetypes, time, sys, etc.) are part of the Python standard library.

Install Flask with:

```
bash
pip install flask
```
# 3. Folder Structure (Expected)
When the application runs, it creates two folders in the same directory as the script:
```
video_reviewer/
├── app.py                     # The single‑file Flask application
├── input_vid/                 # Place videos to review here
└── reviewed_vid/              # Reviewed videos are moved here (auto‑created)
```
No additional folders are required. The input_vid folder must be populated with video files before starting the server.

# 4. Architecture & Design
## 4.1. Overview
The application follows a classic Flask pattern:

* Routes handle page rendering, file moves, and video streaming.

* Helper functions manage file operations, path security, and retry logic.

* HTML template is embedded as a multi-line string and rendered with render_template_string.

## 4.2. Key Components
| Component | Description |
| :--- | :--- |
| `INDEX_HTML` | Bootstrap UI with two columns, embedded video players, action buttons, and flash messages. |
| `get_video_list()` | Scans a folder and returns sorted list of video files (based on extension). |
| `safe_path()`	| Prevents directory traversal by combining secure_filename and commonpath checks. |
| `move_with_retry()` |	Attempts to move a file; retries up to 5 times with exponential backoff if file is locked. |
| `open_video_file_for_streaming()`	| Opens a file with FILE_SHARE_DELETE on Windows (allows move while open). |
| `/`	| Main page – renders the UI with current video lists. |
| `/review/<filename>` | Moves a video from input_vid to reviewed_vid. |
| `/revert/<filename>` | Moves a video from reviewed_vid back to input_vid. |
| `/video/<folder>/<filename>` | Streams a video with HTTP Range support; uses the special file-open method. |
## 4.3. Why a Single File?
Keeping the application in a single file simplifies deployment, modification, and version control. The embedded HTML string can be easily extracted into a separate templates/ folder if the project grows.

5. Code Walkthrough
5.1. Imports & Configuration
Flask, render_template_string, request, redirect, url_for, flash, abort, send_file, Response – core Flask components.

secure_filename from Werkzeug – sanitises user-supplied filenames.

mimetypes – guesses MIME type for video files.

shutil – moves files.

time – used in retry delays.

sys – detects Windows platform.

Constants:

INPUT_FOLDER, REVIEWED_FOLDER – directory names.

VIDEO_EXTENSIONS – supported file extensions.

5.2. Helper Functions
is_video_file(filename)
Checks extension against VIDEO_EXTENSIONS.

get_video_list(folder)
Returns sorted list of video files in the given folder.

safe_path(base_dir, filename)
Ensures the resolved absolute path is still inside base_dir. Uses os.path.commonpath.

move_with_retry(src, dst, max_retries=5, delay=0.5)
Attempts shutil.move; on OSError (e.g. Windows error 32) it waits and retries with doubled delay. Raises the final exception if all attempts fail.

open_video_file_for_streaming(file_path)
Windows: opens with os.O_RDONLY | os.O_BINARY and, if available, os.O_SHARE_DELETE. This allows the file to be renamed/moved while the handle is still open.

Other platforms: normal open().

5.3. Routes
@app.route('/')
Calls get_video_list for both folders, renders INDEX_HTML with the two lists.

@app.route('/review/<path:filename>')
Builds source path in INPUT_FOLDER, destination in REVIEWED_FOLDER.

Calls move_with_retry.

Flashes success or error message.

Redirects to index.

@app.route('/revert/<path:filename>')
Symmetrical to /review/, but moves from REVIEWED_FOLDER to INPUT_FOLDER.

@app.route('/video/<folder>/<path:filename>')
Validates folder is either 'input_vid' or 'reviewed_vid'.

Resolves file path with safe_path.

Determines file size and MIME type.

If no Range header, serves the entire file with send_file.

If Range header exists, parses start and optional end byte, then streams a 206 Partial Content response.

Uses the file handle returned by open_video_file_for_streaming() to allow concurrent move operations.

5.4. HTML Template (INDEX_HTML)
Bootstrap 5 layout, two columns.

Each video is shown inside a .video-card with its filename, a <video> tag, and an action button.

Action buttons trigger a JavaScript confirmation dialog before navigating.

Flash messages are displayed as dismissible Bootstrap alerts.

A small hint warns users to stop video playback before clicking the button.

6. Running the Application
Place video files in the input_vid/ folder (create it if it doesn’t exist).

Run the script:

bash
python app.py
Open a web browser at http://localhost:5000.

The server listens on all interfaces (0.0.0.0) by default. To change the port, modify app.run(port=...).

7. Key Features Explained
7.1. HTTP Range (Seekable Video)
The /video/... endpoint parses the Range header and returns 206 Partial Content. This allows browsers to seek forward/backward in the video and load only the required byte ranges. The implementation is lightweight and compatible with all modern browsers.

7.2. “File in Use” Resilience
On Windows, browsers may keep a handle open on a video file while it is being played. The application counters this with two strategies:

Retry logic – if a move fails due to a locked file, it tries again after a short delay.

FILE_SHARE_DELETE flag – while streaming, the file is opened with sharing permissions that allow deletion/renaming. This flag is available on Windows (via os.O_SHARE_DELETE).

These two mechanisms together make it highly unlikely that a user will encounter a “file in use” error, even if they click the button while the video is playing.

7.3. Security
secure_filename sanitises user-supplied filenames, removing path separators and other dangerous characters.

safe_path verifies that the resolved absolute path is still inside the intended base folder, preventing directory traversal attacks.

8. API Reference
Endpoint	Method	Description
/	GET	Renders main page.
/review/<filename>	GET	Moves file from input_vid to reviewed_vid. Redirects to /.
/revert/<filename>	GET	Moves file from reviewed_vid to input_vid. Redirects to /.
/video/<folder>/<filename>	GET	Streams video file with support for Range header. folder must be input_vid or reviewed_vid.
All endpoints expect filename to be URL-encoded (Flask handles this automatically).

9. Customization & Configuration
9.1. Changing Folder Names
Edit INPUT_FOLDER and REVIEWED_FOLDER at the top of the script.

9.2. Adding/Removing Video Formats
Modify the VIDEO_EXTENSIONS set. Example:

python
VIDEO_EXTENSIONS = {'.mp4', '.webm', '.mov', '.avi', '.mkv', '.flv'}
9.3. Adjusting Retry Behaviour
Change the max_retries and delay parameters in the calls to move_with_retry() inside the route functions.

9.4. Switching to External Template
To move the HTML to a separate file:

Create a templates/ folder.

Save the INDEX_HTML string as index.html inside it.

Replace render_template_string(INDEX_HTML, ...) with render_template('index.html', ...).

Import render_template from Flask.

10. Future Development Ideas
The following enhancements are planned or could be implemented:

10.1. User Authentication
Add login pages and user roles (e.g., reviewer, admin) to restrict access.

10.2. Database Backend
Replace filesystem scanning with a SQLite/PostgreSQL database to store video metadata, review status, timestamps, and user comments.

10.3. Batch Operations
Allow selecting multiple videos and marking them as reviewed / revert all at once.

10.4. Search & Filter
Add a search bar and filter by filename, date, or custom tags.

10.5. Video Thumbnails
Generate thumbnails using ffmpeg or OpenCV and display them instead of the full video player (improves performance for large libraries).

10.6. Progress Indicator
Show a spinner while moving large files, or use AJAX to avoid full page reloads.

10.7. Configuration File
Move folder paths, retry settings, and video extensions into a separate config.py or .env file.

10.8. Unit Tests
Write tests using pytest and Flask’s test client to verify file operations and streaming.

10.9. Docker Deployment
Provide a Dockerfile for containerised deployment.

10.10. Asynchronous Processing
Use asyncio or a task queue (Celery) for moving large files without blocking the web server.

11. Troubleshooting
Problem	Likely Cause	Solution
[WinError 32] after retries	File is still open in another process (e.g., media player, antivirus).	Stop the video playback, wait a moment, try again. Use the retry hint in the UI.
Videos do not play / only show black screen	Incorrect MIME type or unsupported codec.	Ensure the file extension is in VIDEO_EXTENSIONS. Try a different browser.
FileNotFoundError when moving	Race condition – file was deleted/moved elsewhere.	Refresh the page and try again.
Page does not load / connection refused	Port 5000 already in use.	Change the port in app.run(port=5001) or stop the other process.
Directory traversal vulnerability warning	safe_path() check failed.	Ensure filenames do not contain ../ or other path sequences. secure_filename removes them.
12. Contributing
To contribute:

Fork the repository.

Create a feature branch.

Make changes, keeping the code style consistent (PEP 8).

Update this documentation if necessary.

Submit a pull request.

All contributions are welcome – bug fixes, new features, or documentation improvements.

13. License
This project is open-source and available under the MIT License. Feel free to use, modify, and distribute it.

Document version 1.0 – February 2026
