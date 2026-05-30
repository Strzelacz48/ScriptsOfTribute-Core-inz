using System.Text;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using ScriptsOfTribute;
using ScriptsOfTribute.AI;
using ScriptsOfTribute.Board;
using ScriptsOfTribute.Serializers;

namespace Bots;

/// Use as: dotnet run --project GameRunner -- DataCollectorBot MaxPrestigeBot -n 5000 -t 4
public class DataCollectorBot : AI
{
    private readonly AI _inner = new MCTSBot();
    private readonly string _outputPath = "training_data.jsonl";
    private readonly List<JObject> _records = new();
    private string _gameId = "";
    private int _turnIndex;
    private int _moveIndex;
    private static readonly object _writeLock = new();
    private static int _gamesCompleted = 0;
    private static int _gamesWon = 0;
    private static readonly System.Diagnostics.Stopwatch _sw = System.Diagnostics.Stopwatch.StartNew();
    private const int ProgressEvery = 10;

    public override void PregamePrepare()
    {
        _gameId = Guid.NewGuid().ToString("N")[..8];
        _turnIndex = 0;
        _moveIndex = 0;
        _records.Clear();
        _inner.Id = Id;
        _inner.PregamePrepare();
    }

    public override PatronId SelectPatron(List<PatronId> availablePatrons, int round)
        => _inner.SelectPatron(availablePatrons, round);

    public override Move Play(GameState gameState, List<Move> possibleMoves, TimeSpan remainingTime)
    {
        var move = _inner.Play(gameState, possibleMoves, remainingTime);

        if (move.Command != CommandEnum.END_TURN)
            _records.Add(BuildRecord(gameState, move));

        if (move.Command == CommandEnum.END_TURN)
        {
            _turnIndex++;
            _moveIndex = 0;
        }
        else
        {
            _moveIndex++;
        }

        return move;
    }

    public override void GameEnd(EndGameState endState, FullGameState? finalBoardState)
    {
        int won = endState.Winner == Id ? 1 : 0;
        foreach (var r in _records)
            r["won"] = won;

        lock (_writeLock)
        {
            using var writer = new StreamWriter(_outputPath, append: true, Encoding.UTF8);
            foreach (var r in _records)
                writer.WriteLine(r.ToString(Formatting.None));
        }

        int completed = Interlocked.Increment(ref _gamesCompleted);
        if (won == 1) Interlocked.Increment(ref _gamesWon);

        if (completed % ProgressEvery == 0)
        {
            double winRate = (double)_gamesWon / completed * 100;
            double elapsed = _sw.Elapsed.TotalSeconds;
            double gamesPerSec = completed / elapsed;
            // \r returns cursor to line start without newline — overwrites previous update in place.
            // Trailing spaces wipe leftover characters if the new line is shorter than the previous one.
            Console.Write($"\r[{completed} games | win {winRate:F1}% | {gamesPerSec:F1} g/s | {elapsed:F0}s]      ");
        }

        _inner.GameEnd(endState, finalBoardState);
    }

    private JObject BuildRecord(GameState gs, Move move)
    {
        var me = gs.CurrentPlayer;
        var en = gs.EnemyPlayer;

        // Patron states relative to me: 1=I have favour, -1=enemy has favour, 0=neutral
        // Use a fixed-size array indexed by PatronId cast to int (excluding TREASURY)
        var patronArr = new JArray();
        foreach (var kv in gs.PatronStates.All)
        {
            if (kv.Key == PatronId.TREASURY) continue;
            int v = kv.Value == me.PlayerID ? 1
                  : kv.Value == PlayerEnum.NO_PLAYER_SELECTED ? 0
                  : -1;
            patronArr.Add(new JObject { ["id"] = (int)kv.Key, ["v"] = v });
        }

        // Encode move: command type + target id (CardId or PatronId cast to int, -1 if none)
        int actType = (int)move.Command;
        var actIds = new JArray();
        switch (move)
        {
            case SimpleCardMove scm:
                actIds.Add((int)scm.Card.CommonId);
                break;
            case SimplePatronMove spm:
                actIds.Add((int)spm.PatronId);
                break;
            case MakeChoiceMoveUniqueCard mcm:
                foreach (var c in mcm.Choices) actIds.Add((int)c.CommonId);
                break;
            case MakeChoiceMoveUniqueEffect mce:
                foreach (var e in mce.Choices) actIds.Add((int)e.ParentCard.CommonId);
                break;
        }

        return new JObject
        {
            // Identity
            ["g"]         = _gameId,
            ["t"]         = _turnIndex,
            ["m"]         = _moveIndex,
            ["won"]       = -1,  // filled retroactively in GameEnd

            // My scalars
            ["me_p"]      = me.Prestige,
            ["me_c"]      = me.Coins,
            ["me_pw"]     = me.Power,
            ["me_pc"]     = (int)me.PatronCalls,

            // Enemy scalars
            ["en_p"]      = en.Prestige,
            ["en_c"]      = en.Coins,
            ["en_pw"]     = en.Power,

            // My card zones (as CardId integer lists)
            ["hand"]      = new JArray(me.Hand.Select(c => (int)c.CommonId)),
            ["draw"]      = new JArray(me.DrawPile.Select(c => (int)c.CommonId)),
            ["cool"]      = new JArray(me.CooldownPile.Select(c => (int)c.CommonId)),
            ["played"]    = new JArray(me.Played.Select(c => (int)c.CommonId)),
            ["known"]     = new JArray(me.KnownUpcomingDraws.Select(c => (int)c.CommonId)),

            // My agents: [[cardId, currentHp], ...]
            ["my_ag"]     = new JArray(me.Agents.Select(a =>
                                new JArray((int)a.RepresentingCard.CommonId, a.CurrentHp))),

            // Enemy card zones (hand+draw merged — we can't see the split)
            ["en_hd"]     = new JArray(en.HandAndDraw.Select(c => (int)c.CommonId)),
            ["en_cool"]   = new JArray(en.CooldownPile.Select(c => (int)c.CommonId)),
            ["en_played"] = new JArray(en.Played.Select(c => (int)c.CommonId)),

            // Enemy agents: [[cardId, currentHp], ...]
            ["en_ag"]     = new JArray(en.Agents.Select(a =>
                                new JArray((int)a.RepresentingCard.CommonId, a.CurrentHp))),

            // Tavern
            ["tv_av"]     = new JArray(gs.TavernAvailableCards.Select(c => (int)c.CommonId)),
            ["tv_rm"]     = new JArray(gs.TavernCards.Select(c => (int)c.CommonId)),

            // Patron states
            ["pat"]       = patronArr,

            // Chosen action
            ["act"]       = actType,
            ["act_ids"]   = actIds,
        };
    }
}
