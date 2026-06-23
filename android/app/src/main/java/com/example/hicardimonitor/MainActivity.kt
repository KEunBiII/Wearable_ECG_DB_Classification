package com.example.hicardimonitor

import android.net.Uri
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.widget.*
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import org.json.JSONArray
import org.json.JSONObject
import java.util.concurrent.Executors
import kotlin.math.roundToInt

class MainActivity : AppCompatActivity() {

    private lateinit var ecgView: EcgView
    private lateinit var fileNameText: TextView
    private lateinit var beatText: TextView
    private lateinit var hrText: TextView
    private lateinit var rhythmText: TextView
    private lateinit var labelText: TextView
    private lateinit var descText: TextView
    private lateinit var statText: TextView
    private lateinit var logText: TextView
    private lateinit var playButton: Button
    private lateinit var resetButton: Button
    private lateinit var speedSeek: SeekBar
    private lateinit var speedText: TextView
    private lateinit var aiStatusText: TextView
    private lateinit var alertText: TextView
    private lateinit var predClassText: TextView
    private lateinit var predDescText: TextView

    private lateinit var classifier: EcgClassifier
    private val inferExecutor = Executors.newSingleThreadExecutor()
    @Volatile private var modelReady = false

    private val handler = Handler(Looper.getMainLooper())
    private var beats: List<Beat> = emptyList()
    private var fs: Float = 250f
    private var recordName: String = "—"
    private var beatIndex = 0
    private var isPlaying = false
    private var speed = 1.0f
    private var lastRPeak: Int? = null

    private val stats = mutableMapOf<String, Int>()

    private val classNames = listOf(
        "Normal", "Sinus_Tachy", "APC", "AF_AFL", "Bradycardia", "VPC", "Trigeminy"
    )

    private val classDesc = mapOf(
        "Normal" to "정상",
        "Sinus_Tachy" to "동성빈맥",
        "APC" to "심방 조기수축",
        "AF_AFL" to "심방세동/조동",
        "Bradycardia" to "서맥",
        "VPC" to "심실 조기수축",
        "Trigeminy" to "삼단맥"
    )

    private val arrhythmiaClasses = classNames.filter { it != "Normal" }

    private val classColors = mapOf(
        "Normal" to "#00ff88",
        "Sinus_Tachy" to "#ff9800",
        "APC" to "#00cfff",
        "AF_AFL" to "#aa44ff",
        "Bradycardia" to "#4488ff",
        "VPC" to "#ff4444",
        "Trigeminy" to "#ff66aa"
    )

    private val jsonPicker = registerForActivityResult(ActivityResultContracts.GetContent()) { uri: Uri? ->
        if (uri != null) loadJsonFromUri(uri)
    }

    private val playRunnable = object : Runnable {
        override fun run() {
            if (!isPlaying) return
            if (beatIndex >= beats.size) {
                isPlaying = false
                playButton.text = "▶ 재생"
                addLog("재생 완료")
                return
            }
            processBeat(beats[beatIndex])
            beatIndex++
            val delay = (1000f / speed).roundToInt().coerceAtLeast(50)
            handler.postDelayed(this, delay.toLong())
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        ecgView = findViewById(R.id.ecgView)
        fileNameText = findViewById(R.id.fileNameText)
        beatText = findViewById(R.id.beatText)
        hrText = findViewById(R.id.hrText)
        rhythmText = findViewById(R.id.rhythmText)
        labelText = findViewById(R.id.labelText)
        descText = findViewById(R.id.descText)
        statText = findViewById(R.id.statText)
        logText = findViewById(R.id.logText)
        playButton = findViewById(R.id.playButton)
        resetButton = findViewById(R.id.resetButton)
        speedSeek = findViewById(R.id.speedSeek)
        speedText = findViewById(R.id.speedText)
        aiStatusText = findViewById(R.id.aiStatusText)
        alertText = findViewById(R.id.alertText)
        predClassText = findViewById(R.id.predClassText)
        predDescText = findViewById(R.id.predDescText)

        classNames.forEach { stats[it] = 0 }

        loadModelAsync()

        findViewById<Button>(R.id.openButton).setOnClickListener {
            jsonPicker.launch("application/json")
        }

        playButton.setOnClickListener { togglePlay() }
        resetButton.setOnClickListener { resetPlayback() }

        speedSeek.max = 19
        speedSeek.progress = 1
        speedSeek.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(seekBar: SeekBar?, progress: Int, fromUser: Boolean) {
                speed = 0.5f + progress * 0.5f
                speedText.text = "${String.format("%.1f", speed)} beat/s"
            }
            override fun onStartTrackingTouch(seekBar: SeekBar?) {}
            override fun onStopTrackingTouch(seekBar: SeekBar?) {}
        })

        updateUiEmpty()
        addLog("JSON 파일을 선택하세요")
    }

    private fun loadJsonFromUri(uri: Uri) {
        try {
            val jsonString = contentResolver.openInputStream(uri)?.bufferedReader()?.use { it.readText() }
                ?: throw IllegalArgumentException("파일을 읽을 수 없습니다")

            val root = JSONObject(jsonString)
            recordName = root.optString("record", "—")
            fs = root.optDouble("fs", 250.0).toFloat()
            ecgView.fs = fs
            beats = parseBeats(root.getJSONArray("beats"))

            fileNameText.text = "Record: $recordName"
            resetPlayback()
            playButton.isEnabled = beats.isNotEmpty()
            resetButton.isEnabled = beats.isNotEmpty()
            addLog("로드 완료: ${beats.size} beat, fs=${fs.roundToInt()}Hz")
        } catch (e: Exception) {
            Toast.makeText(this, "JSON 로드 오류: ${e.message}", Toast.LENGTH_LONG).show()
            addLog("JSON 로드 오류: ${e.message}")
        }
    }

    private fun parseBeats(array: JSONArray): List<Beat> {
        val result = mutableListOf<Beat>()
        for (i in 0 until array.length()) {
            val obj = array.getJSONObject(i)
            val waveformJson = obj.optJSONArray("waveform") ?: JSONArray()
            val labelsJson = obj.optJSONArray("labels") ?: JSONArray()

            val waveform = MutableList(waveformJson.length()) { idx ->
                waveformJson.optDouble(idx, 0.0).toFloat()
            }
            val labels = MutableList(classNames.size) { idx ->
                labelsJson.optDouble(idx, 0.0).toFloat()
            }
            val rPeak = if (obj.has("r_peak")) obj.optInt("r_peak") else null
            result.add(Beat(waveform, labels, rPeak))
        }
        return result
    }

    private fun togglePlay() {
        if (beats.isEmpty()) return
        isPlaying = !isPlaying
        if (isPlaying) {
            playButton.text = "⏸ 일시정지"
            addLog("재생 시작")
            handler.post(playRunnable)
        } else {
            playButton.text = "▶ 재생"
            addLog("일시정지")
        }
    }

    private fun resetPlayback() {
        isPlaying = false
        handler.removeCallbacks(playRunnable)
        beatIndex = 0
        lastRPeak = null
        classNames.forEach { stats[it] = 0 }
        ecgView.clear()
        playButton.text = "▶ 재생"
        predClassText.text = "—"
        predClassText.setTextColor(android.graphics.Color.parseColor("#00ff88"))
        predDescText.text = if (modelReady) "비트마다 AI가 예측합니다" else "모델 없음 (label 모드)"
        predDescText.setTextColor(android.graphics.Color.parseColor("#5a7a9a"))
        alertText.visibility = android.view.View.GONE
        updateUiEmpty()
        addLog("처음으로 이동")
    }

    override fun onDestroy() {
        super.onDestroy()
        try { classifier.close() } catch (_: Throwable) {}
        inferExecutor.shutdown()
    }

    private fun processBeat(beat: Beat) {
        ecgView.addWaveform(beat.waveform)

        val trueClasses = beat.labels.mapIndexedNotNull { index, value ->
            if (value > 0.5f) classNames.getOrNull(index) else null
        }.ifEmpty { listOf("Normal") }

        trueClasses.forEach { cls -> stats[cls] = (stats[cls] ?: 0) + 1 }

        val hr = calculateHr(beat)
        updateResultPanel(trueClasses, hr)

        if (modelReady) runInference(beat.waveform)
    }

    private fun loadModelAsync() {
        classifier = EcgClassifier(this)
        aiStatusText.text = "모델 로딩 중..."
        inferExecutor.execute {
            val ok = classifier.load()
            handler.post {
                modelReady = ok
                if (ok) {
                    aiStatusText.text = "✅ 모델 로드됨"
                    aiStatusText.setTextColor(android.graphics.Color.parseColor("#4caf50"))
                    addLog("AI 모델 로드 완료 (ExecuTorch)")
                } else {
                    aiStatusText.text = "⚠ 모델 없음 (label 모드)"
                    aiStatusText.setTextColor(android.graphics.Color.parseColor("#ff9800"))
                    predDescText.text = "모델 로드 실패: ${classifier.lastError ?: "알 수 없는 오류"}"
                    addLog("AI 모델 로드 실패: ${classifier.lastError}")
                }
            }
        }
    }

    private fun runInference(waveform: List<Float>) {
        inferExecutor.execute {
            val probs = classifier.classify(waveform) ?: return@execute
            var top = 0
            for (i in probs.indices) if (probs[i] > probs[top]) top = i
            val cls = classNames.getOrElse(top) { "Normal" }
            val p = probs[top]
            handler.post { showPrediction(cls, p) }
        }
    }

    private fun showPrediction(cls: String, prob: Float) {
        val color = android.graphics.Color.parseColor(classColors[cls] ?: "#00ff88")
        val desc = classDesc[cls] ?: cls
        predClassText.setTextColor(color)
        predClassText.text = cls.replace("_", " ")
        predDescText.setTextColor(color)
        predDescText.text = "$desc   ${"%.1f".format(prob * 100)}%"

        if (cls in arrhythmiaClasses) {
            alertText.visibility = android.view.View.VISIBLE
            alertText.text = "⚠ ${cls.replace("_", " ")} 감지 ($desc)  ${"%.1f".format(prob * 100)}%"
        } else {
            alertText.visibility = android.view.View.GONE
        }
    }

    private fun calculateHr(beat: Beat): Int? {
        val currentR = beat.rPeak
        if (currentR != null && lastRPeak != null && currentR > lastRPeak!!) {
            val rrSamples = currentR - lastRPeak!!
            lastRPeak = currentR
            val rrSec = rrSamples / fs
            return if (rrSec > 0f) (60f / rrSec).roundToInt().coerceIn(20, 250) else null
        }
        if (currentR != null) lastRPeak = currentR

        // r_peak가 절대 샘플 위치가 아닌 JSON이면 fallback 사용
        return if (beat.waveform.isNotEmpty()) {
            val sec = beat.waveform.size / fs
            if (sec > 0f) (60f / sec).roundToInt().coerceIn(20, 250) else null
        } else null
    }

    private fun updateResultPanel(trueClasses: List<String>, hr: Int?) {
        val first = trueClasses.first()
        val desc = trueClasses.joinToString(" / ") { classDesc[it] ?: it }

        beatText.text = "Beat: $beatIndex / ${beats.size}"
        labelText.text = trueClasses.joinToString(", ")
        descText.text = desc

        if (hr != null) {
            hrText.text = "$hr bpm"
            rhythmText.text = when {
                hr > 100 -> "Tachycardia / 빠른 맥박"
                hr < 60 -> "Bradycardia / 느린 맥박"
                else -> "Normal Rhythm / 정상 리듬"
            }
        } else {
            hrText.text = "— bpm"
            rhythmText.text = "리듬 계산 대기"
        }

        val total = beatIndex + 1
        val arrCount = arrhythmiaClasses.sumOf { stats[it] ?: 0 }
        val arrPct = if (total > 0) arrCount * 100f / total else 0f
        val topArr = arrhythmiaClasses.maxByOrNull { stats[it] ?: 0 } ?: "Normal"
        val topCnt = stats[topArr] ?: 0

        val verdict = when {
            arrPct >= 20f -> "중증 부정맥 의심"
            arrPct >= 5f -> "중등도 부정맥 의심"
            arrPct >= 1f -> "경증 부정맥 의심"
            else -> "정상 범위"
        }

        statText.text = "전체 ${total} beat | 부정맥 ${arrCount}개 (${String.format("%.1f", arrPct)}%)\n" +
                "주요 감지: ${if (topCnt > 0) topArr else "없음"}\n" +
                "종합 소견: $verdict"
    }

    private fun updateUiEmpty() {
        beatText.text = "Beat: 0 / ${beats.size}"
        hrText.text = "— bpm"
        rhythmText.text = "리듬 계산 대기"
        labelText.text = "—"
        descText.text = "JSON 파일을 열고 재생하세요"
        statText.text = "전체 0 beat | 부정맥 0개 (0.0%)\n주요 감지: 없음\n종합 소견: 분석 대기"
        playButton.isEnabled = beats.isNotEmpty()
        resetButton.isEnabled = beats.isNotEmpty()
    }

    private fun addLog(message: String) {
        logText.append("$message\n")
    }
}

data class Beat(
    val waveform: List<Float>,
    val labels: List<Float>,
    val rPeak: Int?
)
