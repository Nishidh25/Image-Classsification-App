package com.nss.imageclasssification.ui.classify;

import android.app.Dialog;
import android.content.Context;
import android.content.DialogInterface;
import android.content.Intent;
import android.content.SharedPreferences;
import android.graphics.Bitmap;
import android.graphics.ImageDecoder;
import android.graphics.RectF;
import android.graphics.drawable.ColorDrawable;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.MotionEvent;
import android.view.Surface;
import android.view.View;
import android.view.ViewGroup;
import android.view.Window;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.RelativeLayout;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.cardview.widget.CardView;
import androidx.fragment.app.Fragment;

import com.google.android.material.floatingactionbutton.ExtendedFloatingActionButton;
import com.google.android.material.floatingactionbutton.FloatingActionButton;
import com.nss.imageclasssification.R;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.gpu.GpuDelegate;
import org.tensorflow.lite.nnapi.NnApiDelegate;
import org.tensorflow.lite.support.common.FileUtil;
import org.tensorflow.lite.support.common.TensorOperator;
import org.tensorflow.lite.support.common.TensorProcessor;
import org.tensorflow.lite.support.common.ops.NormalizeOp;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;
import org.tensorflow.lite.support.image.ops.ResizeWithCropOrPadOp;
import org.tensorflow.lite.support.image.ops.Rot90Op;
import org.tensorflow.lite.support.label.TensorLabel;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.net.URI;
import java.nio.MappedByteBuffer;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.PriorityQueue;

import static android.app.Activity.RESULT_CANCELED;
import static android.content.ContentValues.TAG;


public class Classification_UI extends Fragment {
    private static final int PICK_IMAGE = 100;
    Uri imageUri;
    //private static final float IMAGE_MEAN = 127.0f;
    //private static final float IMAGE_STD = 128.0f;
    //private static final float PROBABILITY_MEAN = 0.0f;
    //private static final float PROBABILITY_STD = 1.0f;

    final float IMAGE_MEAN = 127.5f;
    final float IMAGE_STD = 127.5f;
    private static final float PROBABILITY_MEAN = 1.0f;
    private static final float PROBABILITY_STD = 1.0f;
    private static final int MAX_RESULTS = 3;

    /**
     * Optional GPU delegate for accleration.
     */
    private GpuDelegate gpuDelegate = null;

    /**
     * Optional NNAPI delegate for accleration.
     */
    private NnApiDelegate nnApiDelegate = null;

    /**
     * An instance of the driver class to run model inference with Tensorflow Lite.
     */
    protected Interpreter tflite;

    /**
     * Options for configuring the Interpreter.
     */
    private final Interpreter.Options tfliteOptions = new Interpreter.Options();


    protected TensorOperator getPreprocessNormalizeOp() {
        Log.d(TAG, "getPreprocessNormalizeOp: " + IMAGE_MEAN + IMAGE_STD);
        return new NormalizeOp(IMAGE_MEAN, IMAGE_STD);
    }

    protected TensorOperator getPostprocessNormalizeOp() {
        Log.d(TAG, "getPostprocessNormalizeOp: "+ PROBABILITY_MEAN + PROBABILITY_STD );
        return new NormalizeOp(PROBABILITY_MEAN, PROBABILITY_STD);
    }

    @Override
    public View onCreateView(@NonNull LayoutInflater inflater,
                              ViewGroup container, Bundle savedInstanceState) {
        View root = inflater.inflate(R.layout.image_classification, container, false);

        final ImageView imageView =  root.findViewById(R.id.imageView);
        final ExtendedFloatingActionButton extendedFab = root.findViewById(R.id.extended_fab);
        extendedFab.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                CardView cardView =(CardView) getActivity().findViewById(R.id.cardView);
                cardView.setVisibility(View.VISIBLE);

                Intent gallery = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.INTERNAL_CONTENT_URI);
                startActivityForResult(gallery, PICK_IMAGE);
                //Intent cameraIntent = new Intent(android.provider.MediaStore.ACTION_IMAGE_CAPTURE);
                //startActivityForResult(cameraIntent, PICK_IMAGE);
            }
        });
        return root;
    }

    protected int getScreenOrientation() {
        switch (getActivity().getWindowManager().getDefaultDisplay().getRotation()) {
            case Surface.ROTATION_270:
                return 270;
            case Surface.ROTATION_180:
                return 180;
            case Surface.ROTATION_90:
                return 90;
            default:
                return 0;
        }
    }

    public static class Recognition {
        /**
         * A unique identifier for what has been recognized. Specific to the class, not the instance of
         * the object.
         */
        private final String id;

        /** Display name for the recognition. */
        private final String title;

        /**
         * A sortable score for how good the recognition is relative to others. Higher should be better.
         */
        private final Float confidence;

        /** Optional location within the source image for the location of the recognized object. */
        private RectF location;

        public Recognition(
                final String id, final String title, final Float confidence, final RectF location) {
            this.id = id;
            this.title = title;
            this.confidence = confidence;
            this.location = location;
        }

        public String getId() {
            return id;
        }

        public String getTitle() {
            return title;
        }

        public Float getConfidence() {
            return confidence;
        }

        public RectF getLocation() {
            return new RectF(location);
        }

        public void setLocation(RectF location) {
            this.location = location;
        }

        @Override
        public String toString() {
            String resultString = "";
            if (id != null) {
                resultString += "[" + id + "] ";
            }

            if (title != null) {
                resultString += title + " ";
            }

            if (confidence != null) {
                resultString += String.format("(%.1f%%) ", confidence * 100.0f);
            }

            if (location != null) {
                resultString += location + " ";
            }

            return resultString.trim();
        }
    }

    private static List<Recognition> getTopKProbability(Map<String, Float> labelProb) {
        // Find the best classifications.
        PriorityQueue<Recognition> pq =
                new PriorityQueue<>(
                        MAX_RESULTS,
                        new Comparator<Recognition>() {
                            @Override
                            public int compare(Recognition lhs, Recognition rhs) {
                                // Intentionally reversed to put high confidence at the head of the queue.
                                return Float.compare(rhs.getConfidence(), lhs.getConfidence());
                            }
                        });

        for (Map.Entry<String, Float> entry : labelProb.entrySet()) {
            pq.add(new Recognition("" + entry.getKey(), entry.getKey(), entry.getValue(), null));
        }

        final ArrayList<Recognition> recognitions = new ArrayList<>();
        int recognitionsSize = Math.min(pq.size(), MAX_RESULTS);
        for (int i = 0; i < recognitionsSize; ++i) {
            recognitions.add(pq.poll());
        }
        return recognitions;
    }


    protected void showResultsInBottomSheet(List<Recognition> results) {
        if (results != null && results.size() >= 3) {
            Recognition recognition = results.get(0);

            if (recognition != null) {
                if (recognition.getTitle() != null) Log.d(TAG, "Title 1 : " + recognition.getTitle()); //recognitionTextView.setText(recognition.getTitle());
                if (recognition.getConfidence() != null)

                    Log.d(TAG, String.format("showResultsInBottomSheet: %.2f",(100 * recognition.getConfidence())));
                    //recognitionValueTextView.setText(String.format("%.2f", (100 * recognition.getConfidence())) + "%");
            }

            Recognition recognition1 = results.get(1);
            if (recognition1 != null) {
                if (recognition1.getTitle() != null) Log.d(TAG, "Title 2 : " + recognition1.getTitle()); //recognition1TextView.setText(recognition1.getTitle());
                if (recognition1.getConfidence() != null)

                    Log.d(TAG, String.format("showResultsInBottomSheet: %.2f",(100 * recognition1.getConfidence())));
                    //recognition1ValueTextView.setText(String.format("%.2f", (100 * recognition1.getConfidence())) + "%");
            }

            Recognition recognition2 = results.get(2);
            if (recognition2 != null) {
                if (recognition2.getTitle() != null) Log.d(TAG, "Title 3 : " + recognition2.getTitle()); //recognition2TextView.setText(recognition2.getTitle());
                if (recognition2.getConfidence() != null)

                    Log.d(TAG, String.format("showResultsInBottomSheet: %.2f",(100 * recognition2.getConfidence())));
                    //recognition2ValueTextView.setText(String.format("%.2f", (100 * recognition2.getConfidence())) + "%");
            }

            View root = getActivity().findViewById(R.id.image_classification).getRootView();
            final TextView textView2 =  root.findViewById(R.id.textView4);
            String Prediction1 = "1. " + recognition.getTitle() +  String.format(Locale.getDefault()," : %.2f",(-100 * recognition.getConfidence())) + "%";
            textView2.setText(Prediction1);


            final TextView textView3 =  root.findViewById(R.id.textView3);
            String Prediction2 = "2. " + recognition1.getTitle() +  String.format(Locale.getDefault()," : %.2f",(-100 * recognition1.getConfidence())) + "%";
            textView3.setText(Prediction2);


            final TextView textView4 =  root.findViewById(R.id.textView2);
            String Prediction3 = "3. " + recognition2.getTitle() +  String.format(Locale.getDefault()," : %.2f",(-100 * recognition2.getConfidence())) + "%";
            textView4.setText(Prediction3);
            //textView4.setId(R.id.textView2);
         /*   for (int i=0; i < 2 ; i++)
            {
                Toast.makeText(getActivity().getApplicationContext(), recognition.getTitle() + " " + (100 * recognition.getConfidence())+ " " + recognition1.getTitle()+ " " + (100 * recognition1.getConfidence()) + " "+ recognition2.getTitle() + " "+ (100 * recognition2.getConfidence()) , Toast.LENGTH_LONG).show();
            }
            Toast toast = Toast.makeText(getActivity().getApplicationContext(), recognition.getTitle() + " " + (100 * recognition.getConfidence())+ " " + recognition1.getTitle()+ " " + (100 * recognition1.getConfidence()) + " "+ recognition2.getTitle() + " "+ (100 * recognition2.getConfidence()) , Toast.LENGTH_LONG);
            toast.show();
        */
        }
    }

    public Uri getImageUri(Bitmap src, Bitmap.CompressFormat format, int quality) {
        ByteArrayOutputStream os = new ByteArrayOutputStream();
        src.compress(format, quality, os);

        String path = MediaStore.Images.Media.insertImage(getActivity().getContentResolver(), src, "title", null);
        return Uri.parse(path);
    }

    public void Classify(Bitmap bitmap) throws IOException {
        final int imageSizeX;
        final int imageSizeY;
        final TensorBuffer outputProbabilityBuffer;
        final TensorProcessor probabilityProcessor;
        Integer sensorOrientation;
        sensorOrientation = 90 - getScreenOrientation();
        MappedByteBuffer tfliteModel = FileUtil.loadMappedFile(getActivity(), "inception_v4_1_default_1.tflite");

        //nnApiDelegate = new NnApiDelegate();
        //tfliteOptions.addDelegate(nnApiDelegate);
        //gpuDelegate = new GpuDelegate();
        //tfliteOptions.addDelegate(gpuDelegate);

        tflite =new Interpreter(tfliteModel, tfliteOptions);

        /**
         * Labels corresponding to the output of the vision model.
         */
        List<String> labels = FileUtil.loadLabels(getActivity(), "labels_without_background.txt");


        int imageTensorIndex = 0;
        int[] imageShape = tflite.getInputTensor(imageTensorIndex).shape(); // {1, height, width, 3}
        imageSizeY = imageShape[1];
        imageSizeX = imageShape[2];
        Log.d(TAG, "imageSizeX/Y: " + imageSizeX + imageSizeY );
        DataType imageDataType = tflite.getInputTensor(imageTensorIndex).dataType();
        int probabilityTensorIndex = 0;
        int[] probabilityShape =
                tflite.getOutputTensor(probabilityTensorIndex).shape(); // {1, NUM_CLASSES}
        DataType probabilityDataType = tflite.getOutputTensor(probabilityTensorIndex).dataType();

        // Creates the input tensor.
        /**
         * Input image TensorBuffer.
         */
        TensorImage inputImageBuffer = new TensorImage(imageDataType);

        // Creates the output tensor and its processor.
        outputProbabilityBuffer = TensorBuffer.createFixedSize(probabilityShape, probabilityDataType);

        // Creates the post processor for the output probability.
        probabilityProcessor = new TensorProcessor.Builder().add(getPostprocessNormalizeOp()).build();
        Log.d(TAG, "Bitmap Config : " + bitmap.getConfig());
        //inputImageBuffer = loadImage(bitmap, sensorOrientation);
        // Didn't work bitmap = convert(bitmap, Bitmap.Config.ARGB_8888);
        bitmap = bitmap.copy(Bitmap.Config.ARGB_8888, false);

        //Testing bitmap conversions
        //imageUri = getImageUri(bitmap, Bitmap.CompressFormat.JPEG , 100);
        //imageView.setImageURI(imageUri);

        inputImageBuffer.load(bitmap);
        Log.d(TAG, "Bitmap Config after : " + bitmap.getConfig());
        int cropSize = Math.min(bitmap.getWidth(), bitmap.getHeight());
        int numRotation = sensorOrientation / 90;
        ImageProcessor imageProcessor =
                new ImageProcessor.Builder()
                        .add(new ResizeWithCropOrPadOp(cropSize, cropSize))
                        .add(new ResizeOp(imageSizeX, imageSizeY, ResizeOp.ResizeMethod.NEAREST_NEIGHBOR))
                        .add(new Rot90Op(numRotation))
                        .add(getPreprocessNormalizeOp())
                        .build();
        inputImageBuffer = imageProcessor.process(inputImageBuffer);

        tflite.run(inputImageBuffer.getBuffer(), outputProbabilityBuffer.getBuffer().rewind());

        Map<String, Float> labeledProbability =
                new TensorLabel(labels, probabilityProcessor.process(outputProbabilityBuffer))
                        .getMapWithFloatValue();

        final List<Recognition> results = getTopKProbability(labeledProbability); // return getTopKProbability(labeledProbability)

        showResultsInBottomSheet(results);
    }

    private View.OnTouchListener mTouchListener =new View.OnTouchListener() {
        @Override
        public boolean onTouch(View v, MotionEvent event) {
            Dialog builder = new Dialog(getContext());
            builder.requestWindowFeature(Window.FEATURE_NO_TITLE);
            builder.getWindow().setBackgroundDrawable(
                    new ColorDrawable(android.graphics.Color.TRANSPARENT));
            builder.setOnDismissListener(new DialogInterface.OnDismissListener() {
                @Override
                public void onDismiss(DialogInterface dialogInterface) {
                    //nothing;
                }
            });
            int a=v.getId();
            //if(R.id.imageView==a)
            //{
            String destinationFilename = android.os.Environment.getExternalStorageDirectory().getPath()+ File.separatorChar+"img_1.jpg";
            Log.d(TAG, "onTouch: "+ destinationFilename);
            SharedPreferences sharedPref = getActivity().getPreferences(Context.MODE_PRIVATE);
            String uristr = sharedPref.getString("Uri 1", "2");
            Uri uri = Uri.parse(uristr);    //path of image
           // }
            //else if(R.id.img_View==a) {
              //  uri = Uri.parse("android.resource://" + getPackageName() + "/drawable/profile"); //path of image
            //}
            ImageView imageView = new ImageView(getContext());
            imageView.setImageURI(uri);                //set the image in dialog popup
            //below code fullfil the requirement of xml layout file for dialog popup

            builder.addContentView(imageView, new RelativeLayout.LayoutParams(
                    ViewGroup.LayoutParams.WRAP_CONTENT,
                    ViewGroup.LayoutParams.WRAP_CONTENT));
            builder.show();
            return false;
        }
    };

    void savefile(Uri sourceuri) {
        String sourceFilename= sourceuri.getPath();
        String destinationFilename = android.os.Environment.getExternalStorageDirectory().getPath()+ File.separatorChar+"img_1.jpg";

        BufferedInputStream bis = null;
        BufferedOutputStream bos = null;

        try {
            bis = new BufferedInputStream(new FileInputStream(sourceFilename));
            bos = new BufferedOutputStream(new FileOutputStream(destinationFilename, false));
            byte[] buf = new byte[1024];
            bis.read(buf);
            do {
                bos.write(buf);
            } while(bis.read(buf) != -1);
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            try {
                if (bis != null) bis.close();
                if (bos != null) bos.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }

    @Override
    public void onActivityResult(int requestCode, int resultCode, Intent data) {

        if (resultCode != RESULT_CANCELED) {
            if (resultCode == getActivity().RESULT_OK && requestCode == PICK_IMAGE) {
                imageUri = data.getData();
                Log.i(TAG, "onActivityResult: " + imageUri);
                View root = getActivity().findViewById(R.id.image_classification).getRootView();
                final ImageView imageView =  root.findViewById(R.id.imageView);
                imageView.setImageURI(imageUri);
                savefile(imageUri);
                SharedPreferences sharedPref = getActivity().getPreferences(Context.MODE_PRIVATE);
                SharedPreferences.Editor editor = sharedPref.edit();
                editor.putString("Uri 1", imageUri.toString());
                editor.commit();
                imageView.setOnTouchListener(mTouchListener);
                ImageDecoder.Source source = ImageDecoder.createSource(getActivity().getContentResolver(), imageUri);
                try {
                    Bitmap bitmap = ImageDecoder.decodeBitmap(source);
                    //Bitmap bitmap = (Bitmap) data.getExtras().get("data");
                    Log.i(TAG, "onActivityResult: bitmap = " + bitmap);
                    Classify(bitmap);
                } catch (IOException e) {
                    e.printStackTrace();
                }
                //getActivity().setContentView(R.layout.image_classification);
            }
        }
    }

}



