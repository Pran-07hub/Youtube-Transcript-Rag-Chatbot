# pyrefly: ignore [missing-import]
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled

def transcript_find(video_id):
    ytt_api = YouTubeTranscriptApi()
    return ytt_api.fetch(video_id, languages = ['en', 'hi'])

def transcript_fetch(video_id):

   transcript = transcript_find(video_id)

   transcript = " ".join([snippet.text for snippet in transcript])

   return transcript

def transcript_fetch_with_time(video_id):

    transcript = transcript_find(video_id)
    script = []
    for snippet in transcript:
        script.append((snippet.text, snippet.start, snippet.duration))

    return script
    
# print(transcript_fetch_with_time('rGny1YTYSug'))