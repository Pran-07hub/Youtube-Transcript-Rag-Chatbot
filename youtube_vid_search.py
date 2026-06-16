import yt_dlp

def search_youtube(query, max_results=10):

    ydl_opts = {
        "quiet": True,
        "extract_flat": True
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:

        search_query = f"ytsearch{max_results}:{query}"

        results = ydl.extract_info(
            search_query,
            download=False
        )

    videos = []

    for entry in results["entries"]:

        videos.append({
            "title": entry.get("title"),
            "video_id": entry.get("id"),
            "url": f"https://youtube.com/watch?v={entry.get('id')}"
        })

    return videos
    
# print(search_youtube("india", max_results=1))
