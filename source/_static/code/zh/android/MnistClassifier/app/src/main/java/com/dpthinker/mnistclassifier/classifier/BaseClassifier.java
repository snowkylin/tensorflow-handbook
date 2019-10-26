package com.dpthinker.mnistclassifier.classifier;

import android.app.Activity;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.util.Log;

import com.dpthinker.mnistclassifier.model.BaseModelConfig;
import com.dpthinker.mnistclassifier.model.ModelConfigFactory;

import org.tensorflow.lite.Interpreter;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.AbstractMap;
import java.util.Comparator;
import java.util.Map;
import java.util.PriorityQueue;

public class BaseClassifier {
    private final static String TAG = "BaseClassifier";
    protected static final int RESULTS_TO_SHOW = 3;
    protected Interpreter mTFLite;

    private String mModelPath = "";

    private int mNumBytesPerChannel;

    private int mDimBatchSize;
    private int mDimPixelSize;

    private int mDimImgWidth;
    private int mDimImgHeight;

    private BaseModelConfig mModelConfig;

    private float[][] mLabelProbArray = new float[1][10];

    protected PriorityQueue<Map.Entry<String, Float>> mSortedLabels =
            new PriorityQueue<>(
                    RESULTS_TO_SHOW,
                    new Comparator<Map.Entry<String, Float>>() {
                        @Override
                        public int compare(Map.Entry<String, Float> o1, Map.Entry<String, Float> o2) {
                            return (o1.getValue()).compareTo(o2.getValue());
                        }
                    });

    private void initConfig(BaseModelConfig config) {
        this.mModelConfig = config;
        this.mNumBytesPerChannel = config.getNumBytesPerChannel();
        this.mDimBatchSize = config.getDimBatchSize();
        this.mDimPixelSize = config.getDimPixelSize();
        this.mDimImgWidth = config.getDimImgWeight();
        this.mDimImgHeight = config.getDimImgHeight();
        this.mModelPath = config.getModelName();
    }

    public BaseClassifier(String modelConfig, Activity activity) throws IOException {
        // init configs for this classifier
        initConfig(ModelConfigFactory.getModelConfig(modelConfig));

        // init interpreter with config parameter
        mTFLite = new Interpreter(loadModelFile(activity));
    }

    /** Memory-map the model file in Assets. */
    private MappedByteBuffer loadModelFile(Activity activity) throws IOException {
        AssetFileDescriptor fileDescriptor = activity.getAssets().openFd(mModelPath);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    /** Writes Image data into a {@code ByteBuffer}. */
    protected ByteBuffer convertBitmapToByteBuffer(Bitmap bitmap) {
        int[] intValues = new int[mDimImgWidth * mDimImgHeight];
        scaleBitmap(bitmap).getPixels(intValues,
                0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());

        ByteBuffer imgData = ByteBuffer.allocateDirect(
                mNumBytesPerChannel * mDimBatchSize * mDimImgWidth * mDimImgHeight * mDimPixelSize);
        imgData.order(ByteOrder.nativeOrder());
        imgData.rewind();

        // Convert the image to floating point.
        int pixel = 0;
        for (int i = 0; i < mDimImgWidth; ++i) {
            for (int j = 0; j < mDimImgHeight; ++j) {
                //final int val = intValues[pixel++];
                int val = intValues[pixel++];
                mModelConfig.addImgValue(imgData, val);
            }
        }
        return imgData;
    }

    public Bitmap scaleBitmap(Bitmap bmp) {
        return Bitmap.createScaledBitmap(bmp, mDimImgWidth, mDimImgHeight, true);
    }

    public String doClassify(Bitmap bitmap) {
        // convert Bitmap to TFLite interpreter readable ByteBuffer
        ByteBuffer imgData = convertBitmapToByteBuffer(bitmap);

        // do run interpreter
        long startTime = System.nanoTime();
        mTFLite.run(imgData, mLabelProbArray);
        long endTime = System.nanoTime();
        Log.i(TAG, String.format("run interpreter cost: %f ms",
                (float)(endTime - startTime)/1000000.0f));

        // generate and return result
        return printTopKLabels();
    }

    /** Prints top-K labels, to be shown in UI as the results. */
    public String printTopKLabels() {
        for (int i = 0; i < 10; i++) {
            mSortedLabels.add(new AbstractMap.SimpleEntry<>(""+i, mLabelProbArray[0][i]));
            if (mSortedLabels.size() > RESULTS_TO_SHOW) {
                mSortedLabels.poll();
            }
        }
        StringBuffer textToShow = new StringBuffer();
        final int size = mSortedLabels.size();
        for (int i = 0; i < size; ++i) {
            Map.Entry<String, Float> label = mSortedLabels.poll();
            textToShow.insert(0, String.format("\n%s   %4.8f",label.getKey(),label.getValue()));
        }
        return textToShow.toString();
    }

}
