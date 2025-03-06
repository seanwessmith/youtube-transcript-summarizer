import { AbortController } from "node-abort-controller";
import { parseStringPromise } from "xml2js";

// Constants
const USER_AGENT =
  "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.83 Safari/537.36,gzip(gfe)";
const DEFAULT_TIMEOUT_MS = 10000;

// Custom Error Class
class YoutubeTranscriptError extends Error {
  constructor(message: string) {
    super(`[YoutubeTranscript] 🚨 ${message}`);
  }
}

// Interface for Transcript Segments
interface TranscriptResponse {
  text: string;
  duration: number;
  offset: number;
  lang: string;
}

/**
 * Decodes HTML entities in a string (e.g., &#39; to ')
 * @param str String with potential HTML entities
 * @returns Decoded string
 */
function decodeHtmlEntities(str: string): string {
  const entities: { [key: string]: string } = {
    "&#39;": "'",
    "&apos;": "'",
    "&quot;": '"',
    "&amp;": "&",
    "&lt;": "<",
    "&gt;": ">",
  };
  return str.replace(
    /&#39;|&apos;|&quot;|&amp;|&lt;|&gt;/g,
    (match) => entities[match] || match
  );
}

/**
 * Fetches and parses a YouTube video transcript
 * @param videoId YouTube video ID (e.g., "dQw4w9WgXcQ")
 * @param lang Optional language code (e.g., "en")
 * @returns Promise resolving to an array of transcript segments
 */
async function fetchTranscript(
  videoId: string,
  lang?: string
): Promise<TranscriptResponse[]> {
  // Step 1: Fetch the video page to get caption track URL
  const videoUrl = `https://www.youtube.com/watch?v=${videoId}`;
  const videoPageResponse = await fetchWithTimeout(videoUrl, {
    headers: { "User-Agent": USER_AGENT },
  });
  const videoPageBody = await videoPageResponse.text();

  // Step 2: Extract caption track URL from player response
  const playerResponseMatch = videoPageBody.match(
    /var ytInitialPlayerResponse = ({.+});/
  );
  if (!playerResponseMatch) {
    throw new YoutubeTranscriptError("Failed to find player response data");
  }
  const playerResponse = JSON.parse(playerResponseMatch[1]);
  const captionTracks =
    playerResponse.captions?.playerCaptionsTracklistRenderer?.captionTracks;
  if (!captionTracks || captionTracks.length === 0) {
    throw new YoutubeTranscriptError(
      `No transcripts available for video ${videoId}`
    );
  }

  // Select the desired language or default to the first track
  const selectedTrack = lang
    ? captionTracks.find((track: any) => track.languageCode === lang) ||
      captionTracks[0]
    : captionTracks[0];
  if (!selectedTrack) {
    throw new YoutubeTranscriptError(
      `Language ${lang} not available for video ${videoId}`
    );
  }

  // Step 3: Fetch the transcript XML
  const transcriptResponse = await fetchWithTimeout(selectedTrack.baseUrl, {
    headers: { "User-Agent": USER_AGENT },
  });
  const transcriptBody = await transcriptResponse.text();

  // Step 4: Parse the XML with xml2js
  const parsedXml = await parseStringPromise(transcriptBody, {
    explicitArray: false,
  });
  const textElements = Array.isArray(parsedXml.transcript.text)
    ? parsedXml.transcript.text
    : [parsedXml.transcript.text].filter(Boolean); // Handle single or array elements

  // Step 5: Transform parsed data into transcript segments
  const transcript: TranscriptResponse[] = [];
  for (const elem of textElements) {
    const start = parseFloat(elem.$.start);
    const dur = parseFloat(elem.$.dur);
    const text = decodeHtmlEntities(elem._ || ""); // Decode HTML entities
    if (!isNaN(start) && !isNaN(dur) && text) {
      transcript.push({
        text,
        duration: dur,
        offset: start,
        lang: selectedTrack.languageCode,
      });
    }
  }

  if (transcript.length === 0) {
    throw new YoutubeTranscriptError(
      `No valid transcript data found for video ${videoId}`
    );
  }

  return transcript;
}

/**
 * Fetches a URL with timeout support
 * @param url URL to fetch
 * @param options Fetch options including headers
 * @returns Promise resolving to the fetch Response object
 */
async function fetchWithTimeout(url: string, options: any = {}): Promise<any> {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), DEFAULT_TIMEOUT_MS);

  try {
    const response = await fetch(url, {
      ...options,
      signal: controller.signal,
    });
    if (!response.ok) {
      throw new Error(`HTTP error: ${response.status}`);
    }
    return response;
  } finally {
    clearTimeout(timeoutId);
  }
}

export { fetchTranscript };
