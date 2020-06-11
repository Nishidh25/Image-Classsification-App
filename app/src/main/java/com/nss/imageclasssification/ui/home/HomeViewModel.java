package com.nss.imageclasssification.ui.home;

import androidx.lifecycle.LiveData;
import androidx.lifecycle.MutableLiveData;
import androidx.lifecycle.ViewModel;

public class HomeViewModel extends ViewModel {

    private MutableLiveData<String> mText;

    public HomeViewModel() {
        mText = new MutableLiveData<>();
        mText.setValue("An android app for image classification.  " +
                "It uses EfficientNet-lite model to classify an image selected from storage classification.");
    }

    public LiveData<String> getText() {
        return mText;
    }
}