"""
Tests for the FastAPI server.
"""

import pytest
from fastapi.testclient import TestClient
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server import app


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)


class TestLibraryEndpoint:
    """Tests for the library view endpoint."""

    def test_library_returns_html(self, client):
        """Test that the library endpoint returns HTML."""
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]

    def test_library_contains_expected_elements(self, client):
        """Test that the library page contains expected elements."""
        response = client.get("/")
        assert response.status_code == 200
        # Check for library page elements
        content = response.text.lower()
        assert "library" in content or "reader" in content


class TestUploadEndpoint:
    """Tests for the upload endpoint."""

    def test_upload_without_file_fails(self, client):
        """Test that uploading without a file fails appropriately."""
        response = client.post("/upload")
        # Should fail with 422 (Unprocessable Entity) due to missing file
        assert response.status_code == 422


class TestStaticAssets:
    """Tests for static asset handling."""

    def test_nonexistent_book_returns_404(self, client):
        """Test that requesting a non-existent book returns 404."""
        response = client.get("/read/nonexistent_book_xyz/0")
        assert response.status_code == 404


class TestHealthCheck:
    """Basic health checks for the server."""

    def test_server_starts(self, client):
        """Test that the server can be started and responds."""
        response = client.get("/")
        assert response.status_code == 200

    def test_server_handles_invalid_routes(self, client):
        """Test that invalid routes return 404."""
        response = client.get("/this/route/does/not/exist")
        assert response.status_code == 404


class TestReadingProgressAPI:
    """Tests for reading progress API endpoints."""

    def test_get_progress_nonexistent_book(self, client):
        """Test getting progress for a book with no saved progress."""
        response = client.get("/api/progress/nonexistent_test_book")
        # Should return default progress
        assert response.status_code == 200
        data = response.json()
        assert "book_id" in data
        assert data["chapter_index"] == 0

    def test_save_and_get_progress(self, client):
        """Test saving and retrieving reading progress."""
        book_id = "test_book_progress_123"

        # Save progress
        progress_data = {
            "chapter_index": 5,
            "scroll_position": 0.45,
            "total_chapters": 10
        }
        save_response = client.post(
            f"/api/progress/{book_id}",
            json=progress_data
        )
        assert save_response.status_code == 200
        assert save_response.json()["status"] == "saved"

        # Get progress
        get_response = client.get(f"/api/progress/{book_id}")
        assert get_response.status_code == 200

        data = get_response.json()
        assert data["chapter_index"] == 5


class TestBookmarksAPI:
    """Tests for bookmarks API endpoints."""

    def test_get_bookmarks_empty(self, client):
        """Test getting bookmarks for a book with none."""
        response = client.get("/api/bookmarks/test_book_no_bookmarks")
        assert response.status_code == 200
        data = response.json()
        assert "bookmarks" in data
        assert data["bookmarks"] == []

    def test_create_bookmark(self, client):
        """Test creating a bookmark."""
        book_id = "test_book_bookmarks_123"

        bookmark_data = {
            "chapter_index": 2,
            "scroll_position": 0.5,
            "title": "Test bookmark",
            "note": "Test note"
        }
        response = client.post(
            f"/api/bookmarks/{book_id}",
            json=bookmark_data
        )
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "created"
        assert "id" in data

    def test_get_bookmarks(self, client):
        """Test getting bookmarks for a book."""
        book_id = "test_book_get_bookmarks"

        # Create a bookmark first
        client.post(
            f"/api/bookmarks/{book_id}",
            json={
                "chapter_index": 0,
                "scroll_position": 0.1,
                "title": "Test",
                "note": ""
            }
        )

        response = client.get(f"/api/bookmarks/{book_id}")
        assert response.status_code == 200

        data = response.json()
        assert "bookmarks" in data
        assert len(data["bookmarks"]) >= 1

    def test_delete_bookmark(self, client):
        """Test deleting a bookmark."""
        book_id = "test_book_delete_bookmark"

        # Create a bookmark
        create_response = client.post(
            f"/api/bookmarks/{book_id}",
            json={
                "chapter_index": 0,
                "scroll_position": 0.1,
                "title": "To be deleted",
                "note": ""
            }
        )
        bookmark_id = create_response.json()["id"]

        # Delete it
        del_url = f"/api/bookmarks/{book_id}/{bookmark_id}"
        delete_response = client.delete(del_url)
        assert delete_response.status_code == 200


class TestHighlightsAPI:
    """Tests for highlights API endpoints."""

    def test_get_highlights_empty(self, client):
        """Test getting highlights for a book with none."""
        response = client.get("/api/highlights/test_book_no_highlights")
        assert response.status_code == 200
        data = response.json()
        assert "highlights" in data
        assert data["highlights"] == []

    def test_create_highlight(self, client):
        """Test creating a highlight."""
        book_id = "test_book_highlights_123"

        highlight_data = {
            "chapter_index": 1,
            "text": "Highlighted text",
            "color": "yellow",
            "start_offset": 10,
            "end_offset": 25
        }
        response = client.post(
            f"/api/highlights/{book_id}",
            json=highlight_data
        )
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "created"
        assert "id" in data

    def test_highlight_colors(self, client):
        """Test different highlight colors."""
        book_id = "test_book_highlight_colors"
        colors = ["yellow", "green", "blue", "pink", "purple"]

        for color in colors:
            response = client.post(
                f"/api/highlights/{book_id}",
                json={
                    "chapter_index": 0,
                    "text": f"Text with {color}",
                    "color": color
                }
            )
            assert response.status_code == 200
            assert response.json()["status"] == "created"

    def test_delete_highlight(self, client):
        """Test deleting a highlight."""
        book_id = "test_book_delete_highlight"

        # Create a highlight
        create_response = client.post(
            f"/api/highlights/{book_id}",
            json={
                "chapter_index": 0,
                "text": "To be deleted",
                "color": "yellow"
            }
        )
        highlight_id = create_response.json()["id"]

        # Delete it
        del_url = f"/api/highlights/{book_id}/{highlight_id}"
        delete_response = client.delete(del_url)
        assert delete_response.status_code == 200

    def test_delete_highlight_not_found(self, client):
        """Test deleting a non-existent highlight returns 404."""
        book_id = "test_book_delete_nonexistent"
        delete_response = client.delete(
            f"/api/highlights/{book_id}/nonexistent_id"
        )
        assert delete_response.status_code == 404

    def test_delete_highlight_removes_from_list(self, client):
        """Test that deleted highlight is removed from get list."""
        import uuid
        book_id = f"test_book_delete_verify_{uuid.uuid4().hex[:8]}"

        # Create two highlights
        resp1 = client.post(
            f"/api/highlights/{book_id}",
            json={"chapter_index": 0, "text": "Text 1", "color": "yellow"}
        )
        id1 = resp1.json()["id"]

        resp2 = client.post(
            f"/api/highlights/{book_id}",
            json={"chapter_index": 0, "text": "Text 2", "color": "green"}
        )
        id2 = resp2.json()["id"]

        # Verify both exist
        list_resp = client.get(f"/api/highlights/{book_id}")
        assert len(list_resp.json()["highlights"]) == 2

        # Delete first highlight
        client.delete(f"/api/highlights/{book_id}/{id1}")

        # Verify only one remains
        list_resp = client.get(f"/api/highlights/{book_id}")
        highlights = list_resp.json()["highlights"]
        assert len(highlights) == 1
        assert highlights[0]["id"] == id2

    def test_update_highlight_color(self, client):
        """Test updating a highlight's color."""
        import uuid
        book_id = f"test_book_update_color_{uuid.uuid4().hex[:8]}"

        # Create a highlight
        create_response = client.post(
            f"/api/highlights/{book_id}",
            json={
                "chapter_index": 0,
                "text": "Color change test",
                "color": "yellow"
            }
        )
        highlight_id = create_response.json()["id"]

        # Update color to green
        update_response = client.put(
            f"/api/highlights/{book_id}/{highlight_id}/color",
            json={"color": "green"}
        )
        assert update_response.status_code == 200
        assert update_response.json()["status"] == "updated"

        # Verify color changed
        get_response = client.get(f"/api/highlights/{book_id}")
        highlights = get_response.json()["highlights"]
        assert len(highlights) == 1
        assert highlights[0]["color"] == "green"

    def test_update_highlight_color_all_colors(self, client):
        """Test updating highlight to all valid colors."""
        import uuid
        book_id = f"test_book_all_colors_{uuid.uuid4().hex[:8]}"

        # Create a highlight
        create_response = client.post(
            f"/api/highlights/{book_id}",
            json={
                "chapter_index": 0,
                "text": "Multi color test",
                "color": "yellow"
            }
        )
        highlight_id = create_response.json()["id"]

        # Test all colors
        for color in ["green", "blue", "pink", "purple", "yellow"]:
            update_response = client.put(
                f"/api/highlights/{book_id}/{highlight_id}/color",
                json={"color": color}
            )
            assert update_response.status_code == 200

            # Verify
            get_response = client.get(f"/api/highlights/{book_id}")
            assert get_response.json()["highlights"][0]["color"] == color

    def test_update_highlight_color_not_found(self, client):
        """Test updating color of non-existent highlight returns 404."""
        book_id = "test_book_color_notfound"
        update_response = client.put(
            f"/api/highlights/{book_id}/nonexistent_id/color",
            json={"color": "green"}
        )
        assert update_response.status_code == 404

    def test_update_highlight_color_invalid(self, client):
        """Test updating to invalid color returns 404."""
        book_id = "test_book_invalid_color"

        # Create a highlight
        create_response = client.post(
            f"/api/highlights/{book_id}",
            json={
                "chapter_index": 0,
                "text": "Invalid color test",
                "color": "yellow"
            }
        )
        highlight_id = create_response.json()["id"]

        # Try invalid color
        update_response = client.put(
            f"/api/highlights/{book_id}/{highlight_id}/color",
            json={"color": "red"}
        )
        # Should return 404 since update_highlight_color returns False
        assert update_response.status_code == 404

        # Verify original color unchanged
        get_response = client.get(f"/api/highlights/{book_id}")
        assert get_response.json()["highlights"][0]["color"] == "yellow"


class TestSearchAPI:
    """Tests for search API endpoints."""

    def test_search_short_query(self, client):
        """Test search with very short query."""
        response = client.get("/api/search?q=a")
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        # Short query returns empty results
        assert data["results"] == []

    def test_search_with_book_filter(self, client):
        """Test search with book filter."""
        response = client.get("/api/search?q=test&book_id=test_book")
        assert response.status_code == 200
        assert "results" in response.json()

    def test_search_history(self, client):
        """Test search history endpoint."""
        response = client.get("/api/search/history")
        assert response.status_code == 200
        data = response.json()
        assert "history" in data
        assert isinstance(data["history"], list)

    def test_clear_search_history(self, client):
        """Test clearing search history."""
        response = client.delete("/api/search/history")
        assert response.status_code == 200


class TestExportAPI:
    """Tests for export API endpoints."""

    def test_export_book_json(self, client):
        """Test exporting book data as JSON."""
        book_id = "test_book_export"

        # Add some data first
        client.post(
            f"/api/bookmarks/{book_id}",
            json={
                "chapter_index": 0,
                "scroll_position": 0.5,
                "title": "Export test",
                "note": "Note"
            }
        )

        response = client.get(f"/api/export/{book_id}?format=json")
        assert response.status_code == 200
        # Response is PlainTextResponse, so parse content
        assert "bookmarks" in response.text

    def test_export_book_markdown(self, client):
        """Test exporting book data as Markdown."""
        book_id = "test_book_export_md"

        response = client.get(f"/api/export/{book_id}?format=markdown")
        assert response.status_code == 200
        assert "Notes and Highlights" in response.text

    def test_export_all(self, client):
        """Test exporting all data."""
        response = client.get("/api/export")
        assert response.status_code == 200
        # Should be valid JSON
        assert "exported_at" in response.text

    def test_export_highlights_json(self, client):
        """Test exporting highlights as JSON returns valid JSON."""
        import json
        import uuid
        book_id = f"test_export_highlights_json_{uuid.uuid4().hex[:8]}"

        # Create highlights
        client.post(
            f"/api/highlights/{book_id}",
            json={
                "chapter_index": 0,
                "text": "First highlighted text",
                "color": "yellow"
            }
        )
        client.post(
            f"/api/highlights/{book_id}",
            json={
                "chapter_index": 1,
                "text": "Second highlighted text",
                "color": "green"
            }
        )

        response = client.get(f"/api/export/{book_id}?format=json")
        assert response.status_code == 200

        # Verify it's valid JSON
        data = json.loads(response.text)
        assert "highlights" in data
        assert len(data["highlights"]) == 2
        assert data["highlights"][0]["text"] == "First highlighted text"
        assert data["highlights"][0]["color"] == "yellow"
        assert data["highlights"][1]["text"] == "Second highlighted text"
        assert data["highlights"][1]["color"] == "green"

    def test_export_highlights_markdown(self, client):
        """Test exporting highlights as Markdown contains highlight text."""
        import uuid
        book_id = f"test_export_highlights_md_{uuid.uuid4().hex[:8]}"

        # Create highlights with different colors
        client.post(
            f"/api/highlights/{book_id}",
            json={
                "chapter_index": 0,
                "text": "Yellow highlight text",
                "color": "yellow"
            }
        )
        client.post(
            f"/api/highlights/{book_id}",
            json={
                "chapter_index": 1,
                "text": "Green highlight text",
                "color": "green",
                "note": "This is a note"
            }
        )

        response = client.get(f"/api/export/{book_id}?format=markdown")
        assert response.status_code == 200

        content = response.text
        # Check markdown structure
        assert "# Notes and Highlights" in content
        assert "## Highlights" in content
        # Check highlight text is included
        assert "Yellow highlight text" in content
        assert "Green highlight text" in content
        # Check color emoji
        assert "🟡" in content  # yellow
        assert "🟢" in content  # green
        # Check note is included
        assert "This is a note" in content

    def test_export_bookmarks_and_highlights_json(self, client):
        """Test exporting both bookmarks and highlights as JSON."""
        import json
        import uuid
        book_id = f"test_export_both_json_{uuid.uuid4().hex[:8]}"

        # Create bookmark
        client.post(
            f"/api/bookmarks/{book_id}",
            json={
                "chapter_index": 0,
                "scroll_position": 0.5,
                "title": "My Bookmark",
                "note": "Bookmark note"
            }
        )

        # Create highlight
        client.post(
            f"/api/highlights/{book_id}",
            json={
                "chapter_index": 1,
                "text": "Important passage",
                "color": "blue"
            }
        )

        response = client.get(f"/api/export/{book_id}?format=json")
        assert response.status_code == 200

        data = json.loads(response.text)
        assert len(data["bookmarks"]) == 1
        assert len(data["highlights"]) == 1
        assert data["bookmarks"][0]["title"] == "My Bookmark"
        assert data["highlights"][0]["text"] == "Important passage"

    def test_export_bookmarks_and_highlights_markdown(self, client):
        """Test exporting both bookmarks and highlights as Markdown."""
        import uuid
        book_id = f"test_export_both_md_{uuid.uuid4().hex[:8]}"

        # Create bookmark
        client.post(
            f"/api/bookmarks/{book_id}",
            json={
                "chapter_index": 2,
                "scroll_position": 0.75,
                "title": "Important Section",
                "note": "Review this later"
            }
        )

        # Create highlight
        client.post(
            f"/api/highlights/{book_id}",
            json={
                "chapter_index": 3,
                "text": "Key concept here",
                "color": "purple"
            }
        )

        response = client.get(f"/api/export/{book_id}?format=markdown")
        assert response.status_code == 200

        content = response.text
        # Check both sections exist
        assert "## Bookmarks" in content
        assert "## Highlights" in content
        # Check bookmark content
        assert "Important Section" in content
        assert "Review this later" in content
        # Check highlight content
        assert "Key concept here" in content
        assert "🟣" in content  # purple

    def test_export_empty_book(self, client):
        """Test exporting a book with no highlights or bookmarks."""
        import json
        import uuid
        book_id = f"test_export_empty_{uuid.uuid4().hex[:8]}"

        # Export JSON
        response = client.get(f"/api/export/{book_id}?format=json")
        assert response.status_code == 200
        data = json.loads(response.text)
        assert data["highlights"] == []
        assert data["bookmarks"] == []

        # Export Markdown
        response = client.get(f"/api/export/{book_id}?format=markdown")
        assert response.status_code == 200
        assert "Notes and Highlights" in response.text

    def test_export_invalid_format(self, client):
        """Test that invalid format returns 400 error."""
        response = client.get("/api/export/test_book?format=pdf")
        assert response.status_code == 400
        assert "Format must be" in response.json()["detail"]

    def test_export_json_content_type(self, client):
        """Test that JSON export returns correct content type."""
        response = client.get("/api/export/test_book?format=json")
        assert response.status_code == 200
        assert "application/json" in response.headers["content-type"]

    def test_export_markdown_content_type(self, client):
        """Test that Markdown export returns correct content type."""
        response = client.get("/api/export/test_book?format=markdown")
        assert response.status_code == 200
        assert "text/markdown" in response.headers["content-type"]

    def test_export_json_has_metadata(self, client):
        """Test that JSON export includes metadata."""
        import json
        import uuid
        book_id = f"test_export_metadata_{uuid.uuid4().hex[:8]}"

        response = client.get(f"/api/export/{book_id}?format=json")
        data = json.loads(response.text)

        assert "book_id" in data
        assert data["book_id"] == book_id
        assert "exported_at" in data

    def test_export_all_colors_markdown(self, client):
        """Test that all highlight colors have correct emojis in Markdown."""
        import uuid
        book_id = f"test_export_colors_{uuid.uuid4().hex[:8]}"

        colors = ["yellow", "green", "blue", "pink", "purple"]
        expected_emojis = ["🟡", "🟢", "🔵", "🔴", "🟣"]

        for color in colors:
            client.post(
                f"/api/highlights/{book_id}",
                json={
                    "chapter_index": 0,
                    "text": f"Text with {color}",
                    "color": color
                }
            )

        response = client.get(f"/api/export/{book_id}?format=markdown")
        content = response.text

        for emoji in expected_emojis:
            assert emoji in content, f"Expected emoji {emoji} not found"
