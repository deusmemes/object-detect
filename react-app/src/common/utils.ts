import {Notification} from "rsuite";

export const base64ToString = (bytes: any): string => {
    const base64 = btoa(
        new Uint8Array(bytes).reduce(
            (data, byte) => data + String.fromCharCode(byte),
            '',
        ),
    );

    return "data:;base64," + bytes;
}

export const sendNotify = (type: string, title: string, text: string) => {
    Notification.open({
            title: title,
            description: text
          });
}