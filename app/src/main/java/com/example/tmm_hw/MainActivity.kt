package com.example.tmm_hw

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import com.example.tmm_hw.camera.CameraManager
import com.example.tmm_hw.databinding.ActivityMainBinding
import com.example.tmm_hw.pipeline.ProcessingPipeline
import com.example.tmm_hw.removal.RemovalMode
import com.example.tmm_hw.ui.OverlayView
import kotlinx.coroutines.*
import org.opencv.android.OpenCVLoader
import org.opencv.core.Mat

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private lateinit var cameraManager: CameraManager
    private val pipeline = ProcessingPipeline()

    private var frameCount = 0
    private var lastFpsTime = System.currentTimeMillis()
    private var currentFps = 0

    companion object {
        private const val CAMERA_PERMISSION_REQUEST = 100
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        // Инициализация OpenCV
        if (!OpenCVLoader.initDebug()) {
            Toast.makeText(this, "OpenCV не загружен!", Toast.LENGTH_LONG).show()
            return
        }

        setupUI()

        if (hasCameraPermission()) {
            startCamera()
        } else {
            requestCameraPermission()
        }
    }

    private fun setupUI() {
        // Переключатель обработки
        binding.switchProcessing.setOnCheckedChangeListener { _, isChecked ->
            pipeline.processingEnabled = isChecked
        }

        binding.btnTestNoise.setOnCheckedChangeListener { _, isChecked ->
            pipeline.testNoiseEnabled = isChecked
        }

        // Выбор режима фильтрации
        binding.radioGroupMode.setOnCheckedChangeListener { _, checkedId ->
            pipeline.currentMode = when (checkedId) {
                R.id.radioMedian    -> RemovalMode.MEDIAN
                R.id.radioBilateral -> RemovalMode.BILATERAL
                R.id.radioInpaint   -> RemovalMode.INPAINT
                else                -> RemovalMode.MEDIAN
            }
        }
    }

    private fun startCamera() {
        cameraManager = CameraManager(
            context = this,
            lifecycleOwner = this,
            previewView = binding.previewView,
            onFrameReady = ::processFrame
        )
        cameraManager.startCamera()
    }

    private fun processFrame(frame: Mat) {
        // Обработка в фоновом потоке
        lifecycleScope.launch(Dispatchers.Default) {
            val result = pipeline.process(frame)
            frame.release()

            // Считаем FPS
            frameCount++
            val now = System.currentTimeMillis()
            if (now - lastFpsTime >= 1000) {
                currentFps = frameCount
                frameCount = 0
                lastFpsTime = now
            }

            // Обновляем UI в главном потоке
            withContext(Dispatchers.Main) {
                binding.overlayView.updateFrame(
                    result.processedFrame,
                    result.detectionResult.mask
                )

                val artifactCount = result.detectionResult.artifactCount
                val ratio = (result.detectionResult.artifactRatio * 100).toInt()
                binding.tvStats.text =
                    "Артефактов: $artifactCount ($ratio%) | FPS: $currentFps | ${result.processingTimeMs}ms"

                result.processedFrame.release()
                result.detectionResult.mask.release()
            }
        }
    }

    private fun hasCameraPermission() =
        ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) ==
                PackageManager.PERMISSION_GRANTED

    private fun requestCameraPermission() {
        ActivityCompat.requestPermissions(
            this,
            arrayOf(Manifest.permission.CAMERA),
            CAMERA_PERMISSION_REQUEST
        )
    }

    override fun onRequestPermissionsResult(
        requestCode: Int, permissions: Array<String>, grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == CAMERA_PERMISSION_REQUEST &&
            grantResults.firstOrNull() == PackageManager.PERMISSION_GRANTED
        ) {
            startCamera()
        } else {
            Toast.makeText(this, "Нужно разрешение на камеру", Toast.LENGTH_LONG).show()
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraManager.stopCamera()
        pipeline.release()
    }
}