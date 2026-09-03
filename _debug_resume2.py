import tempfile, json
from pathlib import Path
from dataset_acquisition.downloader import Downloader
from dataset_acquisition.models import Person, SearchResult
from tests.test_dataset_acquisition.test_acquisition import MockSource, MockFaceServicePerImage, _make_person

with tempfile.TemporaryDirectory() as tmpdir:
    output_dir = Path(tmpdir)
    persons = [_make_person("p1")]
    results = [
        SearchResult(source="mock", source_url="http://a1", image_url="http://a1/single.jpg"),
        SearchResult(source="mock", source_url="http://a2", image_url="http://a2/noface.jpg"),
    ]

    # First run
    source = MockSource(results)
    face_service = MockFaceServicePerImage(face_counts=[1, 0])
    dl = Downloader(output_dir=output_dir, sources=[source], max_images_per_person=5, face_service=face_service)
    records1, rej1 = dl.download_person(persons[0])
    print(f"Run1: records={len(records1)}, rej={len(rej1)}")

    # Second run - simulate what download_person does
    source2 = MockSource(results)
    dl2 = Downloader(output_dir=output_dir, sources=[source2], max_images_per_person=5, face_service=MockFaceServicePerImage(face_counts=[1, 0]))
    state = dl2._load_state()
    print(f"Loaded state keys: {list(state.keys())}")
    print(f"rejected_urls raw: {state.get('rejected_urls', {})}")

    person = persons[0]
    existing_source_urls = set(state.get("downloaded", {}).get(person.person_id, []))
    print(f"existing_source_urls: {existing_source_urls}")

    rejected_urls = set(state.get("rejected_urls", {}).get(person.person_id, []))
    print(f"rejected_urls set: {rejected_urls}")
    print(f"rejected_urls type: {type(rejected_urls)}")

    downloaded_ids = existing_source_urls.copy()
    print(f"downloaded_ids: {downloaded_ids}")

    # Simulate the loop
    for result in results:
        print(f"\nChecking result: {result.source_url}")
        print(f"  in downloaded_ids: {result.source_url in downloaded_ids}")
        print(f"  in rejected_urls: {result.source_url in rejected_urls}")
        if result.source_url in downloaded_ids:
            print(f"  -> SKIP (downloaded)")
            continue
        if result.source_url in rejected_urls:
            print(f"  -> SKIP (rejected)")
            continue
        print(f"  -> WOULD PROCESS")
