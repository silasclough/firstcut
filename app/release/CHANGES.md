# What's new in FirstCut

The newest entry is what the update card shows. Keep each line short and in plain words.

## 0.6.0
- It follows your notes. FirstCut reads what you asked for (include this, leave that out, keep it to 3 minutes, no B-roll while he prays), checks each one against the plan, repairs what is missing, and checks the finished video again. If an instruction could mean two places, it asks you first.
- A plan card shows what it understood before it renders (turn on "Show me the plan first" under Options): each instruction with the words it came from, the parts in play order, and buttons to leave a part out or use another take.
- A B-roll shot's own sound never plays unless you ask for it, and sound that grades as poor is flagged with its time.
- B-roll is chosen by what is said: it appears when the speaker names something you can film, and the speaker stays on screen when nothing fits.
- Jump cuts are covered by a punch-in, a camera change or B-roll, and multicam switching follows the speaker.
- Pauses are shortened, not removed, and cuts never land inside a word. Repeated takes keep the best one.
- Graphics are placed away from faces and captions. Numbers, names, lists, quotes and Bible references become cards, and captions stay on for the whole video and stay readable.
- One music ducking curve keeps the music under the voice. Cuts are clean, the sound lands at about -14 LUFS, and the peak stays under -1 dB.
- Final Cut gets a fuller project: roles, markers, notes on each B-roll clip, alternate takes, editable titles, and the same ducking keyframes. It is checked against Final Cut's own format rules, but opening it in Final Cut is still the real test.
- Learn a style from a link: paste a YouTube video, short or channel (or drop a video file) under "How it edits", and FirstCut writes down how it is cut. Save it as a style, add more videos or your own rules, and use it for short or long form.
- The transcript editor opens in the video's order, so the first line is at the top. Switch to "All footage" for the full list, and use Find to search.
- Short clips are graded on hook, flow and value, start just before the first word and end just after the last.
- Better picture quality: no drop in quality after 90 seconds, and a stuck video encoder is stopped and retried.

## 0.5.0
- Recap reels: Short form has a second way in. Give it your finished long video (and a B-roll folder) and it makes 3 vertical reels of about 30 seconds, one for each part of the event. The talk from your video plays underneath, you see the speaker for a moment at the start and once in the middle, and the rest is quick B-roll with captions and music. Shots filmed at 50 or 60 fps play in slow motion at 30 fps (about three in four of them). A logo in the B-roll folder goes in the corner; there are no other graphics. A finished long video has a Make recap reels button.
- Collections: name a list of shots in Find footage and find it again.
- Find footage shows the top 3 keywords for each shot.
- Montage and slideshow videos: a folder with no one talking (or mostly photos) is cut to music, and you check the plan card before it renders.
- The B-roll panel in the live view is smaller, and shots play through once instead of looping.
- B-roll is more varied: the same shot or look is not used again and again.

## 0.4.0
- Cataloguing adapts to you: it uses the whole Mac when you're away, backs off while you work or play a shot, and pauses while another app (a game) needs the Mac. Settings › Footage library says how much it may use. It was about three times faster than 0.2 even before this.
- B-roll Edit: drag the start and end handles of any B-roll shot, slide it frame by frame, and play just that part. Replace, Edit and "Don't use this" no longer re-render by themselves: the video shows "out of date" with a Re-render button (and Discard changes).
- Asking for a second video while one is being made puts it in the queue.
- Dragging in: a video from Photos now comes in as the real video (it used to be a picture), the tab you're on always wins, and several videos build up as thumbnails you can click to preview.
- Short clips open large over the page when you click them, and full screen no longer crops the top and bottom.
- Music: Find my downloads and Import new find songs licensed through Artlist, Soundstripe and similar in Downloads (never YouTube rips) and copy them into your music library, with their licence line kept for your credits.
- Storage: remove old videos after 7, 14, 30 or 60 days (off by default). The plan and every cut are kept, and any project can be re-rendered later.
- Transitions are rare: a fade out at the end, and a fade in only when the video opens on a picture. No cross dissolves.
- How it edits: short clips and long videos each have their own notes and defaults, with a clear switch.
- The transcript shows words not in the video in purple and works for "Add B-roll to a video" projects; the usage bar shows when it resets.
- Progress by time: the bar follows how long each stage usually takes on your Mac, shows how long a step has run and how much time is left, and says when something is taking longer than usual. The YouTube chapters are written after the video is done, so a slow AI answer no longer holds the video at 97%.
- Stop really stops: a frozen render is ended for good a few seconds after you press it.
- The example text in the settings no longer holds anyone's real names, and the app is now built from the same public source that is on GitHub.

## 0.3.0
- Edit from the transcript: scroll the words of a long video, see which are in it (green), cut words out or add them back, then render the changes. While a video is still being made the edit waits for it to finish.
- While a video is being made, the frames it looks at and the B-roll it picks can be clicked. "Don't use this" rules a shot out and it picks something else.
- Speak instead of type, with the words appearing as you talk.
- Find footage: the heat map and "Best part" now follow the camera: the smooth, purposeful move (a push-in, a pan) counts, hand-held settling at the start and people moving in the shot don't.
- Cataloguing reads three files at once and at low priority, so it is about three times faster and stays out of the way of playing shots.
- The logo card is fitted to its card and kept off faces (it was cut off). The hook title in the Final Cut export is the same card as in the app, with an editable copy switched off under it.
- "Add B-roll to a video" projects are named from what is said in them, not the file name.
- The quality check says "No issues found" when there are none, and its details work for older videos too.

## 0.2.1
- Shots in Find footage play straight from the original file, like Quick Look does (no waiting for a copy). Only a file the page can't show falls back to a small copy, made once.
- The quality check opens inside the app, in plain words, with a button that sends its findings back through the editor for a second pass.
- Finding footage skips finished videos: anything in a Final, Exports or Delivered folder, or named like an export (final, V2, rough cut, master). Camera folders are never judged by file name.
- A shot you left (or closed) while its copy was still being prepared no longer starts playing in the background afterwards.
- Footage from log cameras is looked at faster: the colour change is applied to the small picture, not to every 4K frame.

## 0.2.0
- Updates itself: new versions download in the background and install when FirstCut isn't busy.
- Better cuts: no voice over the wrong person, no half-second flashes, B-roll over jump cuts, steadier sound.
- Replace any B-roll shot, and search your whole library for a better one.
- Get the transcript of any video, offline.
- Pace: Automatic, Fast-paced cuts, or Take it slow.
- Log footage (S-Log and friends) comes out in normal colour.
- Works offline with the model on your Mac; uses Claude and ChatGPT together if you have both.
- Music page with waveforms you can scrub, and a player bar.
- Watch your footage library being catalogued, and search it as it goes.

## 0.1.0
- First public build.
