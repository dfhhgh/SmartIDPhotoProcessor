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
    source = MockSource(results)
    face_service = MockFaceServicePerImage(face_counts=[1, 0])
    dl = Downloader(output_dir=output_dir, sources=[source], max_images_per_person=5, face_service=face_service)
    records1, rej1 = dl.download_person(persons[0])
    print(f"Run1: records={len(records1)}, rej={len(rej1)}")

    state = dl._load_state()
    print(f"State rejected_urls: {json.dumps(state.get('rejected_urls', {}))}")
    print(f"State rejection_details count: {len(state.get('rejection_details', {}).get('p1', []))}")

    source2 = MockSource(results)
    dl2 = Downloader(output_dir=output_dir, sources=[source2], max_images_per_person=5, face_service=MockFaceServicePerImage(face_counts=[1, 0]))
    state2 = dl2._load_state()
    print(f"dl2 loaded rejected_urls: {json.dumps(state2.get('rejected_urls', {}))}")
    records2, rej2 = dl2.download_person(persons[0])
    print(f"Run2: records={len(records2)}, rej={len(rej2)}")
    for r in rej2:
        print(f"  rej: {r.source_url} -> {r.rejection_reason}")
