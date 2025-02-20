// FLAC形式の音声ファイルを読み込み、OpenALを使って再生するためのコードです。
// 具体的には、dr_flacというライブラリでFLACファイルをデコードし、その音声データをOpenALの
// バッファに取り込み、再生する仕組みになっています。適宜メモリ開放することで動作を安定させています。

#define _USE_MATH_DEFINES
#define DR_FLAC_IMPLEMENTATION
#include <windows.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <limits.h>
#include "dr_flac.h"
#include <AL/al.h>
#include <AL/alc.h>
#include <AL/alext.h>
#include <locale.h>
#pragma comment(lib, "OpenAL32.lib")
#pragma warning(disable:4996) // fopenの警告を無視

// OpenALのリソース
ALCdevice* device = NULL;
ALCcontext* context = NULL;
ALuint buffer = 0, source = 0;
short* pSampleData = NULL;

void check_al_error(const char* context) {
    ALenum err = alGetError();
    if (err != AL_NO_ERROR) {
        wprintf(L"%hs: OpenAL Error: %hs\n", context, alGetString(err));
    }
}

void check_alc_error(const char* context) {
    ALCenum err = alcGetError(device);
    if (err != ALC_NO_ERROR) {
        wprintf(L"%hs: OpenAL Context Error: %hs\n", context, alcGetString(device, err));
    }
}

// 音声ファイルの読み込み
__declspec(dllexport) void load_audio(const char* file_path) {
    setlocale(LC_ALL, "ja_JP.UTF-8");
    FILE* log_file = fopen("processing_log.txt", "w");
    if (!log_file) {
        wprintf(L"ログファイルを開けません。\n");
        return;
    }
    fprintf(log_file, "ファイルパス: %s\n", file_path);
    fflush(log_file);

    // FLACファイルを読み込む
    drflac* pFlac = drflac_open_file(file_path, NULL);
    if (pFlac == NULL) {
        fprintf(log_file, "FLACファイルを開けない\n");
        fclose(log_file);
        return;
    }
    fprintf(log_file, "FLACファイルの読み込みに成功しました。\n");
    fflush(log_file);

    // オーディオデータを読み込む
    drflac_uint64 totalFrameCount = pFlac->totalPCMFrameCount;
    drflac_uint64 totalSampleCount = totalFrameCount * pFlac->channels;
    fprintf(log_file, "totalFrameCount: %llu\n", totalFrameCount);
    fprintf(log_file, "totalSampleCount: %llu\n", totalSampleCount);
    fflush(log_file);
    if (totalSampleCount > (drflac_uint64)INT_MAX) {
        fprintf(log_file, "サンプル数が多すぎて変換できません。\n");
        drflac_close(pFlac);
        fclose(log_file);
        return;
    }
    pSampleData = (short*)malloc((size_t)totalSampleCount * sizeof(short));
    if (pSampleData == NULL) {
        fprintf(log_file, "メモリを確保できない\n");
        drflac_close(pFlac);
        fclose(log_file);
        return;
    }
    fprintf(log_file, "メモリ確保に成功しました。\n");
    fflush(log_file);
    drflac_uint64 framesRead = drflac_read_pcm_frames_s16(pFlac, totalFrameCount, pSampleData);
    if (framesRead != totalFrameCount) {
        fprintf(log_file, "データの読み込みに失敗しました\n");
        drflac_close(pFlac);
        free(pSampleData);
        fclose(log_file);
        return;
    }
    fprintf(log_file, "FLACファイルのデータ読み込みに成功しました。\n");
    fflush(log_file);

    // オーディオデータの形式を確認してログに出力
    fprintf(log_file, "サンプルレート: %u\n", pFlac->sampleRate);
    fprintf(log_file, "チャネル数: %u\n", pFlac->channels);
    fflush(log_file);

    // OpenALの初期化
    device = alcOpenDevice(NULL);
    if (!device) {
        fprintf(log_file, "OpenALデバイスの初期化に失敗しました。\n");
        free(pSampleData);
        drflac_close(pFlac);
        fclose(log_file);
        return;
    }
    check_alc_error("alcOpenDevice");

    context = alcCreateContext(device, NULL);
    if (!context) {
        fprintf(log_file, "OpenALコンテキストの作成に失敗しました。\n");
        alcCloseDevice(device);
        free(pSampleData);
        drflac_close(pFlac);
        fclose(log_file);
        return;
    }
    check_alc_error("alcCreateContext");

    if (!alcMakeContextCurrent(context)) {
        fprintf(log_file, "OpenALコンテキストの設定に失敗しました。\n");
        alcDestroyContext(context);
        alcCloseDevice(device);
        free(pSampleData);
        drflac_close(pFlac);
        fclose(log_file);
        return;
    }
    check_alc_error("alcMakeContextCurrent");

    alGenBuffers(1, &buffer);
    if (alGetError() != AL_NO_ERROR) {
        fprintf(log_file, "OpenALバッファの生成に失敗しました。\n");
        alcMakeContextCurrent(NULL);
        alcDestroyContext(context);
        alcCloseDevice(device);
        free(pSampleData);
        drflac_close(pFlac);
        fclose(log_file);
        return;
    }
    check_al_error("alGenBuffers");

    alGenSources(1, &source);
    if (alGetError() != AL_NO_ERROR) {
        fprintf(log_file, "OpenALソースの生成に失敗しました。\n");
        alDeleteBuffers(1, &buffer);
        alcMakeContextCurrent(NULL);
        alcDestroyContext(context);
        alcCloseDevice(device);
        free(pSampleData);
        drflac_close(pFlac);
        fclose(log_file);
        return;
    }
    check_al_error("alGenSources");

    ALenum format = (pFlac->channels == 2) ? AL_FORMAT_STEREO16 : AL_FORMAT_MONO16;
    alBufferData(buffer, format, pSampleData, (ALsizei)totalSampleCount * sizeof(short), (ALsizei)pFlac->sampleRate);
    if (alGetError() != AL_NO_ERROR) {
        fprintf(log_file, "OpenALバッファデータの設定に失敗しました。\n");
        alDeleteSources(1, &source);
        alDeleteBuffers(1, &buffer);
        alcMakeContextCurrent(NULL);
        alcDestroyContext(context);
        alcCloseDevice(device);
        free(pSampleData);
        drflac_close(pFlac);
        fclose(log_file);
        return;
    }
    check_al_error("alBufferData");

    alSourcei(source, AL_BUFFER, buffer);
    check_al_error("alSourcei");

    // HRTFの適用
    if (alcIsExtensionPresent(device, "ALC_SOFT_HRTF")) {
        fprintf(log_file, "HRTFをサポートしています。\n");
    }
    else {
        fprintf(log_file, "HRTFをサポートしていません。\n");
    }

    fprintf(log_file, "OpenALとオーディオデータの初期化に成功しました。\n");
    drflac_close(pFlac);
    fclose(log_file);
}

// 音声の再生
__declspec(dllexport) void play_audio() {
    alSourcePlay(source);
    check_al_error("alSourcePlay");
}

// 音声の停止
__declspec(dllexport) void stop_audio() {
    alSourceStop(source);
    check_al_error("alSourceStop");
}

// DLLリソースの解放
__declspec(dllexport) void cleanup_audio() {
    if (source != 0) {
        alDeleteSources(1, &source);
        check_al_error("alDeleteSources");
    }
    if (buffer != 0) {
        alDeleteBuffers(1, &buffer);
        check_al_error("alDeleteBuffers");
    }
    if (context != NULL) {
        alcMakeContextCurrent(NULL);
        check_alc_error("alcMakeContextCurrent(NULL)");
        alcDestroyContext(context);
        check_alc_error("alcDestroyContext");
    }
    if (device != NULL) {
        alcCloseDevice(device);
        check_alc_error("alcCloseDevice");
    }
    if (pSampleData != NULL) {
        free(pSampleData);
    }
}
