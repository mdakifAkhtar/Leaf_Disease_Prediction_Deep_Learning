"""
================================================================================
Leaf Disease Prediction Web Application - Flask Backend
================================================================================
Author  : Mohammad Akif Akhtar
Purpose : Serve a TensorFlow/Keras CNN model that classifies crop leaf images
          into 15 disease/healthy categories, and render a premium, modern
          UI that displays predictions, confidence, disease information,
          treatment guidance, and similar sample images from the dataset.
================================================================================
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import os
import random
import logging
import tempfile
import base64
import mimetypes

import numpy as np
from flask import (
    Flask,
    render_template,
    request,
    jsonify,
    flash,
    redirect,
    url_for,
    send_from_directory,
)

import tensorflow as tf
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image


# ==============================================================================
# CONFIGURATION
# ==============================================================================

class Config:
    """Central configuration for the Flask application."""

    # --- Hardcoded values (per project requirements) -------------------------
    MODEL_PATH = "leaf_disease_model.keras"
    DATASET_FOLDER = "dataset"

    # NOTE: Uploaded images are NOT persisted to disk. Each upload is
    # processed entirely in memory (via a short-lived temporary file that
    # is deleted immediately after prediction), and the image is returned
    # to the browser as a base64 data URI for preview instead of a saved
    # file path. This keeps the server stateless and avoids accumulating
    # user-uploaded images on disk.

    # --- Configurable values --------------------------------------------------
    ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}
    IMAGE_SIZE = (128, 128)          # Target size used during model training
    MAX_CONTENT_LENGTH = 8 * 1024 * 1024  # 8 MB max upload size
    NUM_SIMILAR_IMAGES = 6
    TOP_K_PREDICTIONS = 3
    SECRET_KEY = "leaf-disease-ai-secret-key-change-in-production"


# ==============================================================================
# FLASK APP INITIALIZATION
# ==============================================================================

app = Flask(__name__)
app.config.from_object(Config)
app.secret_key = Config.SECRET_KEY

# Ensure required folders exist at startup
os.makedirs(Config.DATASET_FOLDER, exist_ok=True)

# --- Logging setup -----------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ==============================================================================
# CLASS NAMES (Model output order MUST match training class indices)
# ==============================================================================

CLASS_NAMES = [
    "Pepper__bell___Bacterial_spot",
    "Pepper__bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Tomato__Target_Spot",
    "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato__Tomato_mosaic_virus",
    "Tomato_healthy",
]


# ==============================================================================
# DISEASE INFORMATION DATABASE
# ==============================================================================
# Each entry contains a human-friendly display name plus agronomic details
# used to populate the "Disease Information Card" on the frontend.

DISEASE_INFO = {
    "Pepper__bell___Bacterial_spot": {
        "display_name": "Pepper Bell - Bacterial Spot",
        "category": "Pepper",
        "description": "A bacterial disease caused by Xanthomonas campestris "
                        "that affects bell pepper plants, creating dark, "
                        "water-soaked lesions on leaves and fruit.",
        "causes": "Caused by the bacterium Xanthomonas campestris pv. "
                   "vesicatoria, spread through rain splash, contaminated "
                   "tools, and infected seeds.",
        "symptoms": "Small, dark brown to black spots with yellow halos on "
                     "leaves; spots may merge causing leaf drop; raised, "
                     "scab-like lesions on fruit.",
        "treatment": "Apply copper-based bactericides; remove and destroy "
                      "infected plant debris; avoid overhead irrigation.",
        "prevention": "Use certified disease-free seeds; practice crop "
                       "rotation; maintain proper plant spacing for airflow; "
                       "avoid working in fields when foliage is wet.",
    },
    "Pepper__bell___healthy": {
        "display_name": "Pepper Bell - Healthy",
        "category": "Pepper",
        "description": "This leaf shows no visible signs of disease and "
                        "appears to be in good physiological health.",
        "causes": "Not applicable - the plant is healthy.",
        "symptoms": "Uniform green color, no lesions, no discoloration, "
                     "normal leaf structure.",
        "treatment": "No treatment required. Continue standard care.",
        "prevention": "Maintain balanced fertilization, proper watering "
                       "schedule, and routine pest monitoring to preserve "
                       "plant health.",
    },
    "Potato___Early_blight": {
        "display_name": "Potato - Early Blight",
        "category": "Potato",
        "description": "A common fungal disease of potato caused by "
                        "Alternaria solani, typically appearing on older, "
                        "lower leaves first.",
        "causes": "Caused by the fungus Alternaria solani; favored by warm "
                   "temperatures, high humidity, and plant stress.",
        "symptoms": "Dark brown spots with concentric rings forming a "
                     "'target-board' pattern, surrounded by yellow tissue.",
        "treatment": "Apply fungicides such as chlorothalonil or mancozeb; "
                      "remove severely infected leaves.",
        "prevention": "Rotate crops every 2-3 years; ensure adequate plant "
                       "nutrition; avoid overhead watering; space plants "
                       "for good air circulation.",
    },
    "Potato___Late_blight": {
        "display_name": "Potato - Late Blight",
        "category": "Potato",
        "description": "A highly destructive disease caused by Phytophthora "
                        "infestans, historically responsible for the Irish "
                        "potato famine.",
        "causes": "Caused by the oomycete pathogen Phytophthora infestans; "
                   "spreads rapidly in cool, wet conditions.",
        "symptoms": "Large, irregular, water-soaked dark green to black "
                     "lesions on leaves; white fungal growth on leaf "
                     "undersides in humid conditions.",
        "treatment": "Apply protectant and systemic fungicides immediately; "
                      "remove and destroy infected plants to prevent spread.",
        "prevention": "Plant resistant varieties; ensure good drainage; "
                       "avoid excessive overhead irrigation; monitor weather "
                       "conditions for blight-favorable periods.",
    },
    "Potato___healthy": {
        "display_name": "Potato - Healthy",
        "category": "Potato",
        "description": "This leaf shows no visible signs of disease and "
                        "appears to be in good physiological health.",
        "causes": "Not applicable - the plant is healthy.",
        "symptoms": "Uniform green color, no lesions, no discoloration, "
                     "normal leaf structure.",
        "treatment": "No treatment required. Continue standard care.",
        "prevention": "Maintain balanced fertilization, proper watering "
                       "schedule, and routine pest monitoring to preserve "
                       "plant health.",
    },
    "Tomato_Bacterial_spot": {
        "display_name": "Tomato - Bacterial Spot",
        "category": "Tomato",
        "description": "A bacterial disease affecting tomato foliage, "
                        "stems, and fruit, caused by several Xanthomonas "
                        "species.",
        "causes": "Caused by Xanthomonas spp.; spreads through splashing "
                   "water, contaminated tools, and infected seedlings.",
        "symptoms": "Small, dark, greasy-looking spots on leaves and "
                     "stems; spots may have yellow halos; fruit develops "
                     "raised scab-like lesions.",
        "treatment": "Apply copper-based bactericides combined with "
                      "mancozeb; remove infected plant material promptly.",
        "prevention": "Use pathogen-free seed; rotate crops; avoid "
                       "overhead irrigation; sanitize tools between uses.",
    },
    "Tomato_Early_blight": {
        "display_name": "Tomato - Early Blight",
        "category": "Tomato",
        "description": "A widespread fungal disease caused by Alternaria "
                        "solani, commonly affecting older tomato leaves "
                        "first.",
        "causes": "Caused by the fungus Alternaria solani; thrives in warm, "
                   "humid weather and on nutrient-stressed plants.",
        "symptoms": "Dark brown spots with concentric target-like rings; "
                     "yellowing around lesions; lower leaves affected "
                     "first before spreading upward.",
        "treatment": "Apply fungicides such as chlorothalonil, copper, or "
                      "mancozeb; remove and destroy infected leaves.",
        "prevention": "Practice crop rotation; mulch soil to prevent "
                       "spore splash; stake plants for airflow; avoid "
                       "overhead watering.",
    },
    "Tomato_Late_blight": {
        "display_name": "Tomato - Late Blight",
        "category": "Tomato",
        "description": "A severe and fast-spreading disease caused by "
                        "Phytophthora infestans, capable of destroying "
                        "entire crops within days.",
        "causes": "Caused by the oomycete Phytophthora infestans; spreads "
                   "rapidly during cool, wet weather.",
        "symptoms": "Large, irregular, water-soaked lesions on leaves that "
                     "turn brown/black; white mold on leaf undersides in "
                     "humid conditions; rapid plant collapse.",
        "treatment": "Apply systemic fungicides immediately upon "
                      "detection; remove and destroy infected plants.",
        "prevention": "Plant resistant varieties; ensure proper spacing "
                       "and drainage; monitor local blight forecasts; "
                       "avoid working with wet foliage.",
    },
    "Tomato_Leaf_Mold": {
        "display_name": "Tomato - Leaf Mold",
        "category": "Tomato",
        "description": "A fungal disease caused by Passalora fulva, most "
                        "common in greenhouse tomatoes with high humidity.",
        "causes": "Caused by the fungus Passalora fulva (formerly "
                   "Fulvia fulva); favored by high humidity and poor "
                   "ventilation.",
        "symptoms": "Pale green to yellow spots on upper leaf surface; "
                     "olive-green to grayish-purple fuzzy mold on the "
                     "underside of leaves.",
        "treatment": "Improve ventilation; apply fungicides labeled for "
                      "leaf mold; remove infected leaves promptly.",
        "prevention": "Reduce humidity through proper spacing and "
                       "ventilation; avoid overhead watering; use "
                       "resistant varieties where available.",
    },
    "Tomato_Septoria_leaf_spot": {
        "display_name": "Tomato - Septoria Leaf Spot",
        "category": "Tomato",
        "description": "A common fungal disease caused by Septoria "
                        "lycopersici, primarily affecting lower leaves.",
        "causes": "Caused by the fungus Septoria lycopersici; spreads via "
                   "water splash and prolonged leaf wetness.",
        "symptoms": "Small, circular spots with dark borders and gray "
                     "centers, often with tiny black specks (pycnidia) "
                     "in the center.",
        "treatment": "Apply fungicides such as chlorothalonil or copper-"
                      "based products; remove infected lower leaves.",
        "prevention": "Rotate crops; mulch to reduce soil splash; avoid "
                       "overhead irrigation; stake plants for airflow.",
    },
    "Tomato_Spider_mites_Two_spotted_spider_mite": {
        "display_name": "Tomato - Two-Spotted Spider Mite",
        "category": "Tomato",
        "description": "A pest infestation caused by Tetranychus urticae, "
                        "a tiny arachnid that feeds on plant sap.",
        "causes": "Caused by infestation of Tetranychus urticae (two-"
                   "spotted spider mite); thrives in hot, dry conditions.",
        "symptoms": "Fine yellow/white stippling on leaves; fine webbing "
                     "on leaf undersides; leaves may bronze and dry out "
                     "in severe infestations.",
        "treatment": "Apply miticides or insecticidal soap; introduce "
                      "predatory mites as biological control.",
        "prevention": "Maintain adequate humidity; avoid drought stress; "
                       "regularly inspect leaf undersides; remove weeds "
                       "that host mites.",
    },
    "Tomato__Target_Spot": {
        "display_name": "Tomato - Target Spot",
        "category": "Tomato",
        "description": "A fungal disease caused by Corynespora "
                        "cassiicola, producing target-like lesions on "
                        "leaves, stems, and fruit.",
        "causes": "Caused by the fungus Corynespora cassiicola; favored "
                   "by warm, humid conditions and dense canopy.",
        "symptoms": "Brown lesions with concentric rings resembling a "
                     "target, often with a yellow halo; can affect stems "
                     "and fruit.",
        "treatment": "Apply broad-spectrum fungicides; improve air "
                      "circulation; remove infected foliage.",
        "prevention": "Practice crop rotation; avoid dense planting; "
                       "manage irrigation to reduce leaf wetness duration.",
    },
    "Tomato__Tomato_YellowLeaf__Curl_Virus": {
        "display_name": "Tomato - Yellow Leaf Curl Virus",
        "category": "Tomato",
        "description": "A viral disease transmitted by whiteflies, causing "
                        "severe stunting and yield loss in tomato plants.",
        "causes": "Caused by Tomato Yellow Leaf Curl Virus (TYLCV), "
                   "transmitted primarily by the whitefly Bemisia tabaci.",
        "symptoms": "Upward curling and yellowing of leaves; stunted "
                     "growth; reduced fruit set and smaller fruit size.",
        "treatment": "No cure once infected; remove and destroy infected "
                      "plants; control whitefly populations with "
                      "insecticides.",
        "prevention": "Use virus-resistant varieties; install insect "
                       "netting; control whitefly vectors early in the "
                       "season.",
    },
    "Tomato__Tomato_mosaic_virus": {
        "display_name": "Tomato - Mosaic Virus",
        "category": "Tomato",
        "description": "A viral disease caused by Tomato Mosaic Virus "
                        "(ToMV), leading to mottled foliage and reduced "
                        "plant vigor.",
        "causes": "Caused by Tomato Mosaic Virus (ToMV); spreads through "
                   "contaminated tools, hands, and infected seed.",
        "symptoms": "Light and dark green mottled pattern on leaves; leaf "
                     "distortion and curling; stunted plant growth.",
        "treatment": "No chemical cure; remove and destroy infected "
                      "plants; disinfect tools and hands regularly.",
        "prevention": "Use certified virus-free seed; practice strict "
                       "sanitation; avoid tobacco product contact before "
                       "handling plants.",
    },
    "Tomato_healthy": {
        "display_name": "Tomato - Healthy",
        "category": "Tomato",
        "description": "This leaf shows no visible signs of disease and "
                        "appears to be in good physiological health.",
        "causes": "Not applicable - the plant is healthy.",
        "symptoms": "Uniform green color, no lesions, no discoloration, "
                     "normal leaf structure.",
        "treatment": "No treatment required. Continue standard care.",
        "prevention": "Maintain balanced fertilization, proper watering "
                       "schedule, and routine pest monitoring to preserve "
                       "plant health.",
    },
}


# ==============================================================================
# MODEL LOADING
# ==============================================================================

model = None  # Global model instance, populated by load_model_once()


def load_model_once():
    """
    Load the trained Keras model into memory exactly once at application
    startup. Loading the model on every request would be extremely slow
    and wasteful, so this function is called a single time when the
    Flask app boots.

    Returns:
        tensorflow.keras.Model: The loaded CNN model, or None if loading
        fails (in which case the app will log the error and respond with
        a clear error message on prediction requests).
    """
    global model
    try:
        if not os.path.exists(Config.MODEL_PATH):
            logger.error("Model file not found at path: %s", Config.MODEL_PATH)
            return None

        logger.info("Loading model from %s ...", Config.MODEL_PATH)
        loaded = load_model(Config.MODEL_PATH)
        logger.info("Model loaded successfully.")
        return loaded

    except Exception as exc:  # noqa: BLE001 - log any load failure clearly
        logger.exception("Failed to load model: %s", exc)
        return None


# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def allowed_file(filename):
    """
    Check whether the uploaded file has an allowed image extension.

    Args:
        filename (str): The original filename of the uploaded file.

    Returns:
        bool: True if the file extension is in ALLOWED_EXTENSIONS.
    """
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in Config.ALLOWED_EXTENSIONS
    )


def bytes_to_data_uri(image_bytes, filename):
    """
    Convert raw uploaded image bytes into a base64-encoded data URI so the
    browser can display the uploaded image WITHOUT the server needing to
    save it to disk.

    Args:
        image_bytes (bytes): Raw bytes of the uploaded image.
        filename (str): Original filename, used to detect the MIME type.

    Returns:
        str: A "data:image/<type>;base64,<...>" URI ready for use directly
        as an <img src="..."> value.
    """
    mime_type, _ = mimetypes.guess_type(filename)
    if not mime_type:
        mime_type = "image/jpeg"  # Sensible fallback for jpg/jpeg/png

    encoded = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def preprocess_image(image_path):
    """
    Preprocess an uploaded leaf image using the EXACT SAME pipeline that
    was used during model training, ensuring consistent predictions.

    Steps:
        1. Load image using TensorFlow/Keras.
        2. Resize image to the model's expected input size (128x128).
        3. Convert image into a NumPy array.
        4. Cast datatype to float32.
        5. Normalize pixel values by dividing by 255.0.
        6. Add a batch dimension using np.expand_dims.

    Args:
        image_path (str): Full filesystem path to the uploaded image.

    Returns:
        numpy.ndarray: A preprocessed image array of shape
        (1, 128, 128, 3), ready to be passed to model.predict().
    """
    # Step 1 & 2: Load and resize image
    target_size = Config.IMAGE_SIZE
    if model is not None:
        try:
            # Automatically use the model's expected input size if available
            input_shape = model.input_shape  # e.g. (None, 128, 128, 3)
            if input_shape and len(input_shape) == 4:
                target_size = (input_shape[1], input_shape[2])
        except Exception:  # noqa: BLE001 - fall back to default size
            target_size = Config.IMAGE_SIZE

    img = keras_image.load_img(image_path, target_size=target_size)

    # Step 3: Convert to NumPy array
    img_array = keras_image.img_to_array(img)

    # Step 4: Convert datatype to float32
    img_array = img_array.astype("float32")

    # Step 5: Normalize pixel values to the [0, 1] range
    img_array = img_array / 255.0

    # Step 6: Add batch dimension -> shape becomes (1, H, W, C)
    processed_image = np.expand_dims(img_array, axis=0)

    return processed_image


def predict_leaf(image_path):

    if model is None:
        raise RuntimeError("Model is not loaded.")

    logger.info("STEP 1")

    processed_image = preprocess_image(image_path)

    logger.info("STEP 2")

    prediction = model(processed_image, training=False).numpy()

    logger.info("STEP 3")

    predicted_index = int(np.argmax(prediction))

    logger.info("STEP 4")

    predicted_class = CLASS_NAMES[predicted_index]

    confidence = float(np.max(prediction) * 100)

    logger.info("STEP 5")

    return predicted_class, confidence, prediction[0]


def get_top3_predictions(raw_predictions):
    """
    Compute the Top-K predictions (default 3) sorted from highest to
    lowest confidence.

    Args:
        raw_predictions (numpy.ndarray): Full probability vector for all
        classes, as returned by predict_leaf().

    Returns:
        list[dict]: A list of dictionaries, each containing "class"
        (display name) and "confidence" (rounded percentage), sorted
        descending by confidence.
    """
    # Get indices sorted by descending probability
    sorted_indices = np.argsort(raw_predictions)[::-1]
    top_k_indices = sorted_indices[: Config.TOP_K_PREDICTIONS]

    top_predictions = []
    for idx in top_k_indices:
        class_key = CLASS_NAMES[idx]
        display_name = DISEASE_INFO.get(class_key, {}).get(
            "display_name", class_key
        )
        confidence_value = round(float(raw_predictions[idx] * 100), 2)
        top_predictions.append(
            {"class": display_name, "confidence": confidence_value}
        )

    return top_predictions


def get_similar_images(predicted_class):
    """
    Randomly select sample images of the predicted disease class from the
    local dataset folder, to be displayed as a reference gallery.

    Args:
        predicted_class (str): The raw class key (matches dataset folder
        name) returned by predict_leaf().

    Returns:
        list[str]: A list of relative URL paths (served via the
        /dataset-image route) pointing to sample images. Returns an empty
        list if the dataset folder does not exist or contains no images.
    """
    class_folder = os.path.join(Config.DATASET_FOLDER, predicted_class)

    if not os.path.isdir(class_folder):
        logger.warning("Dataset folder not found: %s", class_folder)
        return []

    # Collect all valid image files in the class folder
    valid_images = [
        f for f in os.listdir(class_folder) if allowed_file(f)
    ]

    if not valid_images:
        return []

    # Randomly sample up to NUM_SIMILAR_IMAGES images (without crashing if
    # the folder has fewer images than requested)
    sample_count = min(Config.NUM_SIMILAR_IMAGES, len(valid_images))
    sampled_images = random.sample(valid_images, sample_count)

    # Build web-accessible URLs via the dedicated dataset-image route
    image_urls = [
        url_for("dataset_image", class_name=predicted_class, filename=img)
        for img in sampled_images
    ]

    return image_urls


def get_disease_information(predicted_class):
    """
    Retrieve detailed disease information (description, causes, symptoms,
    treatment, prevention) for the predicted class.

    Args:
        predicted_class (str): The raw class key returned by
        predict_leaf().

    Returns:
        dict: Disease information dictionary. Falls back to a generic
        placeholder if the class is somehow not found in DISEASE_INFO.
    """
    default_info = {
        "display_name": predicted_class,
        "category": "Unknown",
        "description": "No information available for this class.",
        "causes": "N/A",
        "symptoms": "N/A",
        "treatment": "N/A",
        "prevention": "N/A",
    }
    return DISEASE_INFO.get(predicted_class, default_info)


def determine_health_status(predicted_class):
    """
    Determine whether the predicted class represents a healthy or
    diseased leaf, based on the presence of "healthy" in the class name.

    Args:
        predicted_class (str): The raw class key returned by
        predict_leaf().

    Returns:
        dict: {"status": "Healthy" | "Diseased", "icon": "🟢" | "🔴",
        "label": full display label}
    """
    if "healthy" in predicted_class.lower():
        return {"status": "Healthy", "icon": "🟢", "label": "🟢 Healthy Leaf"}
    return {"status": "Diseased", "icon": "🔴", "label": "🔴 Diseased Leaf"}


# ==============================================================================
# ROUTES
# ==============================================================================

@app.route("/")
def home():
    """Render the home page with the upload interface."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """
    Handle image upload and return a full prediction payload as JSON,
    including predicted class, confidence, top-3 predictions, disease
    information, health status, and similar reference images.
    """
    try:
        # --- Validate that a file was actually sent -------------------------
        if "file" not in request.files:
            return jsonify({"success": False, "error": "No file part in the request."}), 400

        uploaded_file = request.files["file"]

        if uploaded_file.filename == "":
            return jsonify({"success": False, "error": "No file selected."}), 400

        if not allowed_file(uploaded_file.filename):
            return jsonify({
                "success": False,
                "error": "Invalid file type. Allowed types: jpg, jpeg, png."
            }), 400

        # --- Read the upload into memory (NEVER written to permanent storage) --
        filename = secure_filename(uploaded_file.filename)
        file_ext = filename.rsplit(".", 1)[1].lower()
        image_bytes = uploaded_file.read()

        logger.info("Received image '%s' (%d bytes) - processing in memory only.",
                    filename, len(image_bytes))

        # --- Ensure the model is available before predicting -----------------
        if model is None:
            return jsonify({
                "success": False,
                "error": "Model is not loaded on the server. Please contact the administrator."
            }), 500

        # --- Write to a short-lived temp file purely so Keras' load_img can
        #     decode it, then delete it immediately after prediction. The
        #     image is never saved into a permanent uploads directory.
        temp_file = tempfile.NamedTemporaryFile(suffix=f".{file_ext}", delete=False)
        try:
            temp_file.write(image_bytes)
            temp_file.close()

            # --- Run prediction -------------------------------------------------
            predicted_class, confidence, raw_predictions = predict_leaf(temp_file.name)
        finally:
            # Guarantee cleanup even if prediction raises an exception
            if os.path.exists(temp_file.name):
                os.remove(temp_file.name)
                logger.info("Temporary file removed: %s", temp_file.name)

        # --- Gather supporting information --------------------------------------
        top3 = get_top3_predictions(raw_predictions)
        similar_images = get_similar_images(predicted_class)
        disease_info = get_disease_information(predicted_class)
        health_status = determine_health_status(predicted_class)

        # --- Encode the image as a base64 data URI for preview purposes,
        #     instead of returning a saved file path -------------------------
        uploaded_image_data_uri = bytes_to_data_uri(image_bytes, filename)

        response_payload = {
            "success": True,
            "uploaded_image": uploaded_image_data_uri,
            "predicted_class": disease_info["display_name"],
            "confidence": round(confidence, 2),
            "top3_predictions": top3,
            "disease_info": disease_info,
            "health_status": health_status,
            "similar_images": similar_images,
        }

        return jsonify(response_payload), 200

    except RuntimeError as model_error:
        logger.exception("Model runtime error: %s", model_error)
        return jsonify({"success": False, "error": str(model_error)}), 500

    except Exception as exc:  # noqa: BLE001 - catch-all for robustness
        logger.exception("Unexpected error during prediction: %s", exc)
        return jsonify({
            "success": False,
            "error": "An unexpected error occurred while processing the image."
        }), 500


@app.route("/dataset-image/<class_name>/<filename>")
def dataset_image(class_name, filename):
    """
    Serve an individual sample image from the local dataset folder for a
    given class. Used to populate the "Similar Leaf Images" gallery.

    Args:
        class_name (str): The disease class folder name.
        filename (str): The specific image filename within that folder.
    """
    class_folder = os.path.join(Config.DATASET_FOLDER, class_name)
    return send_from_directory(class_folder, filename)


@app.route("/about")
def about():
    """Render the About page."""
    return render_template("index.html", scroll_to="about")


@app.route("/how-it-works")
def how_it_works():
    """Render the How It Works section (handled client-side via anchor)."""
    return render_template("index.html", scroll_to="how-it-works")


@app.route("/contact")
def contact():
    """Render the Contact section (handled client-side via anchor)."""
    return render_template("index.html", scroll_to="contact")


@app.errorhandler(404)
def page_not_found(_error):
    """Custom 404 handler."""
    return render_template("index.html"), 404


@app.errorhandler(413)
def file_too_large(_error):
    """Handle uploads that exceed MAX_CONTENT_LENGTH."""
    flash("File is too large. Maximum upload size is 8 MB.")
    return redirect(url_for("home"))


@app.errorhandler(500)
def internal_server_error(_error):
    """Custom 500 handler."""
    return jsonify({"success": False, "error": "Internal server error."}), 500


# ==============================================================================
# APPLICATION ENTRY POINT
# ==============================================================================

# ==============================================================================
# LOAD MODEL WHEN APP STARTS
# ==============================================================================

# ==============================================================================
# LOAD MODEL WHEN APPLICATION STARTS
# ==============================================================================
# This runs when Gunicorn imports app.py (Render) and also when running
# python app.py locally.

model = load_model_once()

if model is None:
    logger.error("Failed to load model.")
else:
    logger.info("Model loaded successfully.")


# ==============================================================================
# APPLICATION ENTRY POINT
# ==============================================================================
# This block runs ONLY when you execute:
# python app.py
# It does NOT run on Render because Render uses:
# gunicorn app:app

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000)),
        debug=True,
    )
