from crewai_tools import YoutubeChannelSearchTool
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound



# class YoutubeChannelSearchTool:
#     def __init__(self, youtube_channel_handle: str):
#         self.handle = youtube_channel_handle

#     def load_video(self, video_id: str):
#         try:
#             # first try to list available transcripts
#             transcripts = YouTubeTranscriptApi.list_transcripts(video_id)
#             # pick English or first available
#             transcript = transcripts.find_transcript(['en']).fetch(preserve_formatting=True)
#             text = " ".join([t['text'] for t in transcript])
#         except TranscriptsDisabled:
#             # subtitles disabled—fallback to video description or skip
#             text = f"[Transcript unavailable for video {video_id}—subtitles disabled]"
#         except NoTranscriptFound:
#             text = f"[No transcript found for video {video_id}]"
#         return text

# Initialize the tool with a specific Youtube channel handle to target your search
yt_tool = YoutubeChannelSearchTool(youtube_channel_handle='@krishnaik06')

