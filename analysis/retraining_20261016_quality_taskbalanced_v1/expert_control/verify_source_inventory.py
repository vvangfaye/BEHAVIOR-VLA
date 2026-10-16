"""Read-only verification of the frozen expert object inventory."""
def verify_original_objects(client, source):
    expected = {x["key"]: x for x in source["objects"] if not x["relative_path"].startswith("meta/")}
    observed = {}
    marker = ""
    while True:
        page = client.bucket.list_objects(prefix=source["prefix"], marker=marker, max_keys=1000)
        for item in page.object_list:
            if item.key in expected:
                observed[item.key] = (item.size, item.etag.strip('"').lower())
        if not page.is_truncated:
            break
        if page.next_marker == marker:
            raise ValueError("Original OSS inventory pagination stalled")
        marker = page.next_marker
    if set(observed) != set(expected):
        raise ValueError("Original training object(s) missing")
    for key, item in expected.items():
        if observed[key] != (item["bytes"], item["etag"].strip('"').lower()):
            raise ValueError("Original training object identity changed: " + item["relative_path"])
    return dict(files=len(expected), bytes=sum(x["bytes"] for x in expected.values()), all_sizes_etags_verified=True)

