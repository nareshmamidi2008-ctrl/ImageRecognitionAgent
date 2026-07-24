 async function uploadImage() {

    const fileInput = document.getElementById("image");
    const file = fileInput.files[0];

    if (!file) {
        alert("Please select an image");
        return;
    }

    // Show preview
    document.getElementById("preview").src = URL.createObjectURL(file);

    const formData = new FormData();
    formData.append("file", file);

    document.getElementById("object").innerHTML = "Detecting...";

    try {

        const response = await fetch("http://127.0.0.1:8000/predict", {
            method: "POST",
            body: formData
        });

        const data = await response.json();

        if (data.detections.length > 0) {

            document.getElementById("object").innerHTML =
                "Object : " + data.detections[0].object;

            document.getElementById("confidence").innerHTML =
                "Confidence : " + data.detections[0].confidence + "%";

            document.getElementById("time").innerHTML =
                "Processing Time : " + data.processing_time + " sec";

        } else {

            document.getElementById("object").innerHTML = "No Object Detected";

            document.getElementById("confidence").innerHTML = "";

            document.getElementById("time").innerHTML = "";

        }

    }
    catch (err) {
        console.log(err);
        alert("Backend is not running.");
    }
}