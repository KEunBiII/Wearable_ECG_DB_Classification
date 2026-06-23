package com.example.hicardimonitor

import android.content.Context
import android.util.Log
import com.facebook.soloader.nativeloader.NativeLoader
import com.facebook.soloader.nativeloader.SystemDelegate
import org.pytorch.executorch.EValue
import org.pytorch.executorch.Module
import org.pytorch.executorch.Tensor
import java.io.File
import kotlin.math.exp

/**
 * ExecuTorch(.pte) 모델로 ECG 1비트를 7클래스로 분류한다.
 *
 * 입력: float32 [1, 1, 501]  (JSON waveform, z-score 정규화된 상태 그대로)
 * 출력: float32 [1, 7]       (raw logits) -> sigmoid -> 확률
 */
class EcgClassifier(private val context: Context) {

    private var module: Module? = null

    companion object {
        const val TAG = "HiCardiClassifier"
        const val INPUT_LEN = 501
        const val NUM_CLASSES = 7
        private var nativeInitialized = false
    }

    private var loggedFirst = false

    /** 모델 로드. 성공 시 true. (백그라운드 스레드에서 호출 권장) */
    fun load(): Boolean {
        return try {
            if (!nativeInitialized) {
                if (!NativeLoader.isInitialized()) {
                    NativeLoader.init(SystemDelegate())
                }
                nativeInitialized = true
            }
            val modelFile = copyAssetToFile("model.pte")
            module = Module.load(modelFile.absolutePath)
            Log.i(TAG, "MODEL_LOAD_OK path=${modelFile.absolutePath}")
            // self-test: 추론 실행 경로 검증 (zero 입력)
            val selfTest = classify(List(INPUT_LEN) { 0f })
            Log.i(TAG, "SELF_TEST infer=" + (if (selfTest != null) "OK len=${selfTest.size}" else "NULL"))
            true
        } catch (e: Throwable) {
            module = null
            lastError = e.message ?: e.toString()
            Log.e(TAG, "MODEL_LOAD_FAIL: $lastError", e)
            false
        }
    }

    var lastError: String? = null
        private set

    val isLoaded: Boolean get() = module != null

    /**
     * waveform(보통 501 샘플) -> 7클래스 확률(sigmoid).
     * 길이가 501이 아니면 자르거나 0으로 패딩한다.
     */
    fun classify(waveform: List<Float>): FloatArray? {
        val m = module ?: return null
        return try {
            val input = FloatArray(INPUT_LEN)
            val n = minOf(waveform.size, INPUT_LEN)
            for (i in 0 until n) input[i] = waveform[i]
            // 나머지는 0.0f 패딩 (기본값)

            val tensor = Tensor.fromBlob(input, longArrayOf(1, 1, INPUT_LEN.toLong()))
            val outputs = m.forward(EValue.from(tensor))
            val logits = outputs[0].toTensor().getDataAsFloatArray()

            // multi-label -> sigmoid
            val probs = FloatArray(logits.size) { i -> 1f / (1f + exp(-logits[i])) }
            if (!loggedFirst) {
                loggedFirst = true
                Log.i(TAG, "FIRST_INFER probs=" + probs.joinToString(",") { "%.3f".format(it) })
            }
            probs
        } catch (e: Throwable) {
            lastError = e.message ?: e.toString()
            null
        }
    }

    fun close() {
        try { module?.destroy() } catch (_: Throwable) {}
        module = null
    }

    private fun copyAssetToFile(name: String): File {
        val outFile = File(context.filesDir, name)
        // 매번 덮어써서 모델 갱신 시에도 반영되도록 (작은 비용)
        context.assets.open(name).use { input ->
            outFile.outputStream().use { output -> input.copyTo(output) }
        }
        return outFile
    }
}
