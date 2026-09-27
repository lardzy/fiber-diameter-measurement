import { Composition, Folder } from "remotion";
import "./fonts";
import { Tutorial, TutorialScene, tutorials } from "./series/Tutorial";
export const RemotionRoot: React.FC = () => (
  <>
    <Folder name="Guided-V2">
      {tutorials.map((video) => (
        <Composition
          key={video.id}
          id={video.id}
          component={Tutorial}
          defaultProps={{ videoId: video.id }}
          durationInFrames={video.duration}
          fps={30}
          width={1920}
          height={1080}
        />
      ))}
    </Folder>
    <Folder name="Guided-Chapters">
      {tutorials.map((video) => (
        <Folder key={video.id} name={`${video.id}-Chapters`}>
          {video.scenes.map((scene, index) => (
            <Composition
              key={scene.id}
              id={`Guide-${scene.id}`}
              component={TutorialScene}
              defaultProps={{ scene, video, index }}
              durationInFrames={scene.duration}
              fps={30}
              width={1920}
              height={1080}
            />
          ))}
        </Folder>
      ))}
    </Folder>
  </>
);
