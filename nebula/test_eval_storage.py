"""Space checks use expanded inputs; selective cache restore never writes weights."""
import io
from pathlib import Path
import tarfile
from types import SimpleNamespace
import zipfile
import pytest
from nebula import eval_storage as storage


def test_storage_counts_expanded_zip_and_checkpoint_not_compressed_size(tmp_path):
    assets=tmp_path/'assets';assets.mkdir();model=tmp_path/'checkpoint';model.mkdir()
    for name in storage.ARCHIVES:
        with zipfile.ZipFile(assets/name,'w',compression=zipfile.ZIP_DEFLATED) as f:f.writestr('asset.bin',b'a'*10000)
    (model/'weight.bin').write_bytes(b'params')
    report=storage.estimate(assets,model,raw_bytes=99,reserve_gib=8)
    assert sum(report['asset_expanded_bytes'].values())==30000
    assert report['required_disk_bytes']==30000+6+99+8*storage.GIB
    available=report['required_disk_bytes']
    storage.check_space(tmp_path,report,disk_usage=lambda p:SimpleNamespace(free=available))
    with pytest.raises(ValueError,match='actual inputs'):
        storage.check_space(tmp_path,report,disk_usage=lambda p:SimpleNamespace(free=available-1))


def archive(path,members):
    with tarfile.open(path,'w:gz') as f:
        for name,kind,value in members:
            info=tarfile.TarInfo(name);info.type=kind;info.size=len(value) if kind==tarfile.REGTYPE else 0
            f.addfile(info,io.BytesIO(value) if kind==tarfile.REGTYPE else None)


def test_tokenizer_only_restore_does_not_extract_base_weights_or_escaping_files(tmp_path):
    path=tmp_path/'cache.tar.gz'
    archive(path,[('../outside',tarfile.REGTYPE,b'bad'),('./'+storage.TOKENIZER,tarfile.REGTYPE,b'tokenizer'),
                  ('openpi-assets/checkpoints/pi05_base/params/weights',tarfile.REGTYPE,b'large-unused-weights')])
    target=tmp_path/'cache';output=storage.restore_tokenizer(path,target)
    assert output.read_bytes()==b'tokenizer'
    assert [p for p in target.rglob('*') if p.is_file()]==[output]
    assert not (tmp_path/'outside').exists()
    with pytest.raises(ValueError,match='fresh'):storage.restore_tokenizer(path,target)


@pytest.mark.parametrize('kind',[tarfile.SYMTYPE,tarfile.DIRTYPE])
def test_tokenizer_must_be_regular_file(tmp_path,kind):
    path=tmp_path/'cache.tar.gz';archive(path,[(storage.TOKENIZER,kind,b'')])
    with pytest.raises(ValueError,match='regular'):storage.restore_tokenizer(path,tmp_path/'cache')


def test_missing_tokenizer_is_not_silently_downloaded(tmp_path):
    path=tmp_path/'cache.tar.gz';archive(path,[('other',tarfile.REGTYPE,b'content')])
    with pytest.raises(ValueError,match='lacks'):storage.restore_tokenizer(path,tmp_path/'cache')
