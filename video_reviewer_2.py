import os
import sys
import time
import mimetypes
import shutil
from flask import (
    Flask, render_template_string, request, redirect,
    url_for, flash, abort, send_file, Response
)
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = 'replace-this-with-a-secret-key'   # Change for production!

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
INPUT_FOLDER = 'input_vid'
REVIEWED_FOLDER = 'reviewed_vid'
DELETED_FOLDER = 'deleted_vid'          # New folder for deleted videos
VIDEO_EXTENSIONS = {'.mp4', '.webm', '.ogg', '.mov', '.avi', '.mkv'}

# Ensure folders exist
os.makedirs(INPUT_FOLDER, exist_ok=True)
os.makedirs(REVIEWED_FOLDER, exist_ok=True)
os.makedirs(DELETED_FOLDER, exist_ok=True)   # Create deleted folder

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def is_video_file(filename):
    ext = os.path.splitext(filename)[1].lower()
    return ext in VIDEO_EXTENSIONS

def get_video_list(folder):
    """Return sorted list of video files in the given folder."""
    if not os.path.exists(folder):
        return []
    files = os.listdir(folder)
    videos = [f for f in files if is_video_file(f)]
    videos.sort()
    return videos

def safe_path(base_dir, filename):
    """Prevent directory traversal."""
    filename = secure_filename(filename)
    full_path = os.path.join(base_dir, filename)
    if os.path.commonpath([os.path.abspath(base_dir), os.path.abspath(full_path)]) != os.path.abspath(base_dir):
        return None
    return full_path

def move_with_retry(src, dst, max_retries=5, delay=0.5):
    """
    Attempt to move a file. If it fails because the file is locked,
    wait and retry with exponential backoff.
    """
    for attempt in range(max_retries):
        try:
            shutil.move(src, dst)
            return True
        except OSError as e:
            if attempt == max_retries - 1:
                raise e
            time.sleep(delay)
            delay *= 2
    return False

def open_video_file_for_streaming(file_path):
    """
    Open a video file in a way that allows it to be moved/deleted while open.
    On Windows this uses FILE_SHARE_DELETE; on other platforms normal open.
    """
    if sys.platform == 'win32':
        share_delete_flag = getattr(os, 'O_SHARE_DELETE', 0)
        fd = os.open(file_path, os.O_RDONLY | os.O_BINARY | share_delete_flag)
        return os.fdopen(fd, 'rb')
    else:
        return open(file_path, 'rb')

# ----------------------------------------------------------------------
# Embedded HTML template (now with Delete button)
# ----------------------------------------------------------------------
INDEX_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Video Review Tool</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body { background-color: #f8f9fa; padding-top: 20px; }
        .video-card { margin-bottom: 25px; background: white; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); padding: 15px; }
        video { width: 100%; height: auto; max-height: 240px; background: black; border-radius: 4px; }
        .btn-action { margin-top: 10px; width: 100%; }
        .btn-delete { margin-top: 10px; width: 100%; background-color: #dc3545; color: white; }
        .flash-messages { margin-bottom: 20px; }
        .hint { font-size: 0.9rem; color: #6c757d; margin-top: 5px; font-style: italic; }
        .button-group { display: flex; gap: 10px; }
        .button-group form { flex: 1; }
    </style>
</head>
<body>
    <div class="container">
        <h1 class="text-center mb-4">Video Footage Review</h1>

        {% with messages = get_flashed_messages(with_categories=true) %}
            {% if messages %}
                <div class="flash-messages">
                    {% for category, message in messages %}
                        <div class="alert alert-{{ 'success' if category == 'success' else 'danger' }} alert-dismissible fade show" role="alert">
                            {{ message }}
                            <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
                        </div>
                    {% endfor %}
                </div>
            {% endif %}
        {% endwith %}

        <div class="row">
            <!-- Left column: To review -->
            <div class="col-md-6">
                <div class="card">
                    <div class="card-header bg-primary text-white">
                        <h4>📁 To Review (input_vid)</h4>
                    </div>
                    <div class="card-body">
                        {% if input_videos %}
                            <p class="hint">⚠️ Stop video playback before clicking any button to avoid file‑lock errors.</p>
                            {% for video in input_videos %}
                                <div class="video-card">
                                    <h5>{{ video }}</h5>
                                    <video controls preload="metadata">
                                        <source src="{{ url_for('serve_video', folder='input_vid', filename=video) }}" type="video/mp4">
                                        Your browser does not support the video tag.
                                    </video>
                                    <div class="button-group">
                                        <a href="{{ url_for('review_video', filename=video) }}" class="btn btn-success btn-action"
                                           onclick="return confirm('Mark this video as reviewed? It will be moved to reviewed_vid.');">
                                            ✅ Keep (Review)
                                        </a>
                                        <a href="{{ url_for('delete_video', filename=video) }}" class="btn btn-danger btn-action"
                                           onclick="return confirm('Are you sure you want to delete this video? It will be moved to deleted_vid and will NOT be shown here.');">
                                            🗑️ Delete
                                        </a>
                                    </div>
                                </div>
                            {% endfor %}
                        {% else %}
                            <p class="text-muted">No videos waiting for review.</p>
                        {% endif %}
                    </div>
                </div>
            </div>

            <!-- Right column: Reviewed videos -->
            <div class="col-md-6">
                <div class="card">
                    <div class="card-header bg-secondary text-white">
                        <h4>📂 Reviewed (reviewed_vid)</h4>
                    </div>
                    <div class="card-body">
                        {% if reviewed_videos %}
                            <p class="hint">⚠️ Stop video playback before reverting to avoid file‑lock errors.</p>
                            {% for video in reviewed_videos %}
                                <div class="video-card">
                                    <h5>{{ video }}</h5>
                                    <video controls preload="metadata">
                                        <source src="{{ url_for('serve_video', folder='reviewed_vid', filename=video) }}" type="video/mp4">
                                        Your browser does not support the video tag.
                                    </video>
                                    <a href="{{ url_for('revert_video', filename=video) }}" class="btn btn-warning btn-action"
                                       onclick="return confirm('Revert this video back to input_vid?');">
                                        ↩️ Revert to Input
                                    </a>
                                </div>
                            {% endfor %}
                        {% else %}
                            <p class="text-muted">No reviewed videos yet.</p>
                        {% endif %}
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
"""

# ----------------------------------------------------------------------
# Flask Routes
# ----------------------------------------------------------------------
@app.route('/')
def index():
    input_videos = get_video_list(INPUT_FOLDER)
    reviewed_videos = get_video_list(REVIEWED_FOLDER)
    return render_template_string(
        INDEX_HTML,
        input_videos=input_videos,
        reviewed_videos=reviewed_videos
    )

@app.route('/review/<path:filename>')
def review_video(filename):
    """Move video from input_vid to reviewed_vid (with retry)."""
    src = safe_path(INPUT_FOLDER, filename)
    if not src or not os.path.isfile(src):
        flash(f'Video "{filename}" not found in input folder.', 'error')
        return redirect(url_for('index'))

    dst = os.path.join(REVIEWED_FOLDER, os.path.basename(src))
    try:
        move_with_retry(src, dst)
        flash(f'"{filename}" moved to reviewed.', 'success')
    except Exception as e:
        flash(f'Error moving file after several attempts: {e}', 'error')
    return redirect(url_for('index'))

@app.route('/delete/<path:filename>')
def delete_video(filename):
    """Move video from input_vid to deleted_vid (with retry)."""
    src = safe_path(INPUT_FOLDER, filename)
    if not src or not os.path.isfile(src):
        flash(f'Video "{filename}" not found in input folder.', 'error')
        return redirect(url_for('index'))

    dst = os.path.join(DELETED_FOLDER, os.path.basename(src))
    try:
        move_with_retry(src, dst)
        flash(f'"{filename}" moved to deleted (not shown).', 'success')
    except Exception as e:
        flash(f'Error moving file after several attempts: {e}', 'error')
    return redirect(url_for('index'))

@app.route('/revert/<path:filename>')
def revert_video(filename):
    """Move video from reviewed_vid back to input_vid (with retry)."""
    src = safe_path(REVIEWED_FOLDER, filename)
    if not src or not os.path.isfile(src):
        flash(f'Video "{filename}" not found in reviewed folder.', 'error')
        return redirect(url_for('index'))

    dst = os.path.join(INPUT_FOLDER, os.path.basename(src))
    try:
        move_with_retry(src, dst)
        flash(f'"{filename}" reverted to input.', 'success')
    except Exception as e:
        flash(f'Error reverting file after several attempts: {e}', 'error')
    return redirect(url_for('index'))

@app.route('/video/<folder>/<path:filename>')
def serve_video(folder, filename):
    """Stream video with support for Range headers and file‑sharing (delete‑safe)."""
    if folder not in ['input_vid', 'reviewed_vid']:
        abort(404)

    base_dir = INPUT_FOLDER if folder == 'input_vid' else REVIEWED_FOLDER
    file_path = safe_path(base_dir, filename)
    if not file_path or not os.path.isfile(file_path):
        abort(404)

    range_header = request.headers.get('Range', None)
    file_size = os.path.getsize(file_path)
    mime_type, _ = mimetypes.guess_type(file_path)
    if not mime_type:
        mime_type = 'video/mp4'

    # Open the file with sharing enabled (Windows) / normal (others)
    try:
        f = open_video_file_for_streaming(file_path)
    except Exception:
        abort(500)

    # No range -> send entire file
    if not range_header:
        return send_file(f, mimetype=mime_type, conditional=True)

    # Parse Range header (bytes=start-)
    try:
        range_str = range_header.strip().split('=')[1]
        start_byte = int(range_str.split('-')[0])
        end_byte = range_str.split('-')[1]
        if end_byte:
            end_byte = int(end_byte)
        else:
            end_byte = file_size - 1
    except:
        f.close()
        abort(416)

    if start_byte >= file_size or end_byte >= file_size:
        f.close()
        abort(416)

    chunk_size = end_byte - start_byte + 1
    f.seek(start_byte)

    def generate():
        try:
            remaining = chunk_size
            while remaining > 0:
                read_size = min(8192, remaining)
                data = f.read(read_size)
                if not data:
                    break
                yield data
                remaining -= len(data)
        finally:
            f.close()

    response = Response(generate(), status=206, mimetype=mime_type)
    response.headers.add('Content-Range', f'bytes {start_byte}-{end_byte}/{file_size}')
    response.headers.add('Content-Length', str(chunk_size))
    response.headers.add('Accept-Ranges', 'bytes')
    return response

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)