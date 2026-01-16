export interface GalleryImage {
  preview_url: string;
  original_url: string;
  score: number;
  metadata?: {
    title?: string;
    description?: string;
    taken_time?: string;
    camera?: string;
  };
}

export interface GalleryResponse {
  items: GalleryImage[];
  next_cursor?: string;
}

export interface SearchRequest {
  query: string;
  limit: number;
}
