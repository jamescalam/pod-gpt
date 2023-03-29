import os
from pytube import YouTube
from pytube.exceptions import RegexMatchError
from tqdm.auto import tqdm
import requests
import json
from pydantic import BaseModel
from typing import Optional, List, Any


class VideoRecord(BaseModel):
    video_id: str
    channel_id: str
    title: str
    published: str
    url: str
    transcript: Optional[str] = None


class Video:
    def __init__(self, video: VideoRecord):
        self.metadata = video
        try:
            self.yt = YouTube(self.metadata.url)
        except RegexMatchError:
            print(f"RegexMatchError for '{self.metadata.url}'. Video object cannot be created.")

    def _download_mp3(self, save_dir: Optional[str] = './temp'):
        itag = None
        # filters out all the files with "mp4" extension
        files = self.yt.streams.filter(only_audio=True)
        for file in files:
            if file.mime_type == 'audio/mp4':
                itag = file.itag
                break
        if itag is None:
            print("No MP3 audio found.")
        # initialize mp3 file stream
        stream = self.yt.streams.get_by_itag(itag)
        # download the audio
        stream.download(output_path=save_dir, filename=f"{self.metadata.video_id}.mp3")
        self.temp_file = os.path.join(save_dir, f"{self.metadata.video_id}.mp3")
        return self.temp_file

    def _transcribe(self, model: Any):
        # now transcribe
        result = model.transcribe(self.temp_file)
        segments = result['segments']
        # join all segments to get single full transcript for video
        text = ''.join([x['text'] for x in segments])
        self.metadata.transcript = text
        # return
        return text
    
    def transcribe_video(self, model: Any):
        filepath = self._download_mp3()
        transcription = self._transcribe(model)
        # delete mp3 file
        os.unlink(filepath)
        return transcription
    
    def __str__(self):
        return f"{dict(self.metadata)}"


class Channel:
    videos: List[VideoRecord] = []

    def __init__(self, channel_id: str, api_key: str):
        self.channel_id = channel_id
        self.api_key = api_key

    def get_videos_info(self, max_results: Optional[int] = None):
        """Method to scrape all videos and their metadata from a channel

        :param max_results: Maximum number of videos to scrape. If None, all videos are scraped.
        """
        params = {
            "key": self.api_key,
            "part": "snippet",
            "channelId": self.channel_id,
            "type": "video",
            "maxResults": 50
        }
        while True:
            hit_limit = False
            print('.', end='')
            # get a page of video results for this channel
            res = requests.get(
                "https://www.googleapis.com/youtube/v3/search",
                params=params
            )
            # loop through and append the video info to the list
            for record in res.json()['items']:
                _id = record['id']['videoId']
                self.videos.append(VideoRecord(
                    video_id=_id,
                    channel_id=record['snippet']['channelId'],
                    title = record['snippet']['title'],
                    published = record['snippet']['publishedAt'],
                    url = f"https://youtu.be/{_id}",
                    transcript = None
                ))
                # check if we have reached limit
                if max_results is not None and len(self.videos) >= max_results:
                    hit_limit = True
                    break
            if 'nextPageToken' not in res.json().keys() or hit_limit:
                # we have reached the end
                break
            # otherwise we move on to next page
            next_page_token = res.json()['nextPageToken']
            params['pageToken'] = next_page_token
        return {"num_videos": len(self.videos)}
    
    def transcribe_videos(self, model: Any):
        """Method to transcribe all videos in a channel
        """
        for i, video_metadata in enumerate(tqdm(self.videos)):
            video = Video(video_metadata)
            transcription = video.transcribe_video(model)
            self.videos[i].transcript = transcription

    def __repr__(self):
        return f"Channel({self.channel_id})"

    def __str__(self):
        return f"Channel({self.channel_id})"

    def get_videos(self):
        return self.videos

    def save(self, filepath: Optional[str] = 'transcripts.jsonl'):
        # save data
        with open(filepath, 'w') as fp:
            for video in self.videos:
                fp.write(json.dumps(dict(video))+'\n')
