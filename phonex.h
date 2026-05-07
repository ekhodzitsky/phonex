#ifndef PHONEX_H
#define PHONEX_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Engine lifecycle */
void* phonex_engine_new(const char* model_dir);
void* phonex_engine_new_with_pool_size(const char* model_dir, size_t pool_size);
void phonex_engine_free(void* engine);

/* Offline transcription */
char* phonex_transcribe_file(void* engine, const char* wav_path);

/* Streaming */
void* phonex_stream_new(const char* model_dir, const char* vad_path);
char* phonex_stream_process_chunk(void* stream, const float* samples, size_t len);
char* phonex_stream_flush(void* stream);
void phonex_stream_reset(void* stream);
void phonex_stream_free(void* stream);

/* Utility */
void phonex_string_free(char* s);

#ifdef __cplusplus
}
#endif

#endif /* PHONEX_H */
