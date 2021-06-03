import React, {useState} from "react";
import {Button, ButtonToolbar, Icon, Panel, Uploader} from "rsuite";
import {FileType} from "rsuite/es/Uploader";
import { Notification } from 'rsuite';
import {base64ToString, sendNotify} from "../../common/utils";
import {HttpService} from "../../services/http";
import {AnalyzeData, Model, ResultData} from "../../common/types";

const styles = {
    lineHeight: '5em',
}

type UploadProps = {
    onSubmit: (files: FileType[]) => void
}

const UploadImage = ({ onSubmit}: UploadProps) => {
    const [files, setFiles] = useState<FileType[]>([]);

    const clear = () => {
        setFiles([]);
    }

    return (
        <Panel shaded header={'Загрузите несколько изображений'}>
            <Uploader
                name={'files'}
                accept={'image/jpeg,image/png,image/gif,image/tiff'}
                fileList={files}
                fileListVisible={true}
                onChange={(list) => setFiles(list)}
                draggable
                multiple
                listType={'picture-text'}
                autoUpload={false}
            >
                <div style={styles}>
                    <Icon icon='camera-retro' size="5x" style={{marginTop: '0.5em'}}/>
                    <p>Нажмите или перетащите</p>
                </div>
            </Uploader>
            <ButtonToolbar style={{marginTop: '2em'}}>
                <Button
                    appearance={'primary'}
                    onClick={() => onSubmit(files)}
                    disabled={files.length === 0}
                >
                    Отправить
                </Button>
                <Button
                    color={'red'}
                    onClick={clear}
                    disabled={files.length === 0}
                >
                    Очистить
                </Button>
            </ButtonToolbar>
        </Panel>
    )
}

export default UploadImage;