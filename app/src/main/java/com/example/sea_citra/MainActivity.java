package com.example.sea_citra;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.media.ThumbnailUtils;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

import com.example.sea_citra.ml.Sea; // Assuming sea.tflite has generated this class

public class MainActivity extends AppCompatActivity {

    Button camera, gallery;
    ImageView imageView;
    TextView result;
    int imageSize = 32;  // Assuming the model expects 32x32 input images

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        // Initialize UI components
        camera = findViewById(R.id.button);
        gallery = findViewById(R.id.button2);
        result = findViewById(R.id.result);
        imageView = findViewById(R.id.imageView);

        // Set up camera button click listener
        camera.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                if (checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
                    // Open camera to capture image
                    Intent cameraIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
                    startActivityForResult(cameraIntent, 3);
                } else {
                    requestPermissions(new String[]{Manifest.permission.CAMERA}, 100);
                }
            }
        });

        // Set up gallery button click listener
        gallery.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                // Open gallery to pick image
                Intent galleryIntent = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
                startActivityForResult(galleryIntent, 1);
            }
        });
    }

    // Method to classify image using the sea.tflite model
    public void classifyImage(Bitmap image) {
        try {
            // Load the TensorFlow Lite model (sea.tflite)
            Sea model = Sea.newInstance(getApplicationContext());

            // Prepare the input buffer for the model
            TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, imageSize, imageSize, 3}, DataType.FLOAT32);
            ByteBuffer byteBuffer = ByteBuffer.allocateDirect(4 * imageSize * imageSize * 3);
            byteBuffer.order(ByteOrder.nativeOrder());

            // Convert bitmap image into pixel data
            int[] intValues = new int[imageSize * imageSize];
            image.getPixels(intValues, 0, image.getWidth(), 0, 0, image.getWidth(), image.getHeight());
            int pixel = 0;

            // Normalize pixel values (0-255) to (0-1) range
            for (int i = 0; i < imageSize; i++) {
                for (int j = 0; j < imageSize; j++) {
                    int val = intValues[pixel++]; // Get RGB values
                    byteBuffer.putFloat(((val >> 16) & 0xFF) * (1.f / 255)); // Red
                    byteBuffer.putFloat(((val >> 8) & 0xFF) * (1.f / 255));  // Green
                    byteBuffer.putFloat((val & 0xFF) * (1.f / 255));         // Blue
                }
            }

            // Load the input data into the model
            inputFeature0.loadBuffer(byteBuffer);

            // Run the model inference
            Sea.Outputs outputs = model.process(inputFeature0);
            TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();

            // Get the output result (confidence scores)
            float[] confidences = outputFeature0.getFloatArray();

            // Find the index with the highest confidence score
            int maxPos = 0;
            float maxConfidence = confidences[0];
            for (int i = 1; i < confidences.length; i++) {
                if (confidences[i] > maxConfidence) {
                    maxConfidence = confidences[i];
                    maxPos = i;
                }
            }

            // Define the classes (labels) corresponding to the model's output
            String[] classes = {"Sea", "Mountain"};
            result.setText(classes[maxPos]);  // Display the classified result

            // Close the model after inference
            model.close();

        } catch (IOException e) {
            // Handle any errors during the model loading or inference
            e.printStackTrace();
        }
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        if (resultCode == RESULT_OK) {
            if (requestCode == 3) {  // From Camera
                Bitmap image = (Bitmap) data.getExtras().get("data");
                int dimension = Math.min(image.getWidth(), image.getHeight());
                image = ThumbnailUtils.extractThumbnail(image, dimension, dimension);
                imageView.setImageBitmap(image);

                // Resize the image to match the model input size (32x32)
                image = Bitmap.createScaledBitmap(image, imageSize, imageSize, false);
                classifyImage(image);  // Classify the captured image
            } else if (requestCode == 1) {  // From Gallery
                Uri dat = data.getData();
                Bitmap image = null;
                try {
                    image = MediaStore.Images.Media.getBitmap(this.getContentResolver(), dat);
                } catch (IOException e) {
                    e.printStackTrace();
                }
                imageView.setImageBitmap(image);

                // Resize the image to match the model input size (32x32)
                image = Bitmap.createScaledBitmap(image, imageSize, imageSize, false);
                classifyImage(image);  // Classify the selected image
            }
        }
        super.onActivityResult(requestCode, resultCode, data);
    }
}
